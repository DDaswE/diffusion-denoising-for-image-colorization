from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10


HORSE_CLASS_INDEX = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert NCHW RGB images in [0, 1] to NCHW grayscale images."""
    red = rgb[:, 0:1]
    green = rgb[:, 1:2]
    blue = rgb[:, 2:3]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def load_cifar10_horses(
    root: Path,
    train_limit: int,
    test_limit: int,
    download: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = CIFAR10(root=str(root), train=True, download=download)
    test = CIFAR10(root=str(root), train=False, download=download)

    def filter_horses(dataset: CIFAR10, limit: int) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(dataset.targets)
        horse_idx = np.where(targets == HORSE_CLASS_INDEX)[0][:limit]
        rgb = dataset.data[horse_idx].astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (0, 3, 1, 2))
        grey = rgb_to_grayscale(rgb)
        return grey.astype(np.float32), rgb.astype(np.float32)

    train_grey, train_rgb = filter_horses(train, train_limit)
    test_grey, test_rgb = filter_horses(test, test_limit)
    return train_grey, train_rgb, test_grey, test_rgb


@dataclass
class DiffusionSchedule:
    timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor

    @classmethod
    def linear(
        cls,
        timesteps: int = 300,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> "DiffusionSchedule":
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return cls(timesteps, betas, alphas, alphas_cumprod)

    def extract(self, values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def add_noise(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(y0)
        sqrt_alpha_bar = self.extract(torch.sqrt(self.alphas_cumprod), t, y0.shape)
        sqrt_one_minus_alpha_bar = self.extract(torch.sqrt(1.0 - self.alphas_cumprod), t, y0.shape)
        yt = sqrt_alpha_bar * y0 + sqrt_one_minus_alpha_bar * noise
        return yt, noise

    def reconstruct_x0(
        self,
        yt: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = self.extract(torch.sqrt(self.alphas_cumprod), t, yt.shape)
        sqrt_one_minus_alpha_bar = self.extract(torch.sqrt(1.0 - self.alphas_cumprod), t, yt.shape)
        return (yt - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar


class Denoiser(nn.Module):
    def __init__(self, num_filters: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.t_emb = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, num_filters * 2),
        )

        self.downconv1 = nn.Sequential(
            nn.Conv2d(4, num_filters, kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
        )
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.finalconv = nn.Conv2d(num_filters, 3, kernel_size, padding=padding)

    def forward(self, yt: torch.Tensor, x_grey: torch.Tensor, t_normalized: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([yt, x_grey], dim=1)

        d1 = self.downconv1(inputs)
        d2 = self.downconv2(d1)
        features = self.rfconv(d2)

        t_vec = self.t_emb(t_normalized.float().view(-1, 1))
        t_vec = t_vec.view(t_vec.size(0), t_vec.size(1), 1, 1)
        features = features + t_vec

        u1 = self.upconv1(features)
        u1 = u1 + d1
        u2 = self.upconv2(u1)
        return self.finalconv(u2)


def make_loader(grey: np.ndarray, rgb: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(grey), torch.from_numpy(rgb))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def train_denoiser(
    model: Denoiser,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for x_grey, y0 in loader:
            x_grey = x_grey.to(device)
            y0 = y0.to(device)
            batch_size = y0.size(0)

            t = torch.randint(0, schedule.timesteps, (batch_size,), device=device).long()
            yt, noise = schedule.add_noise(y0, t)
            noise_pred = model(yt, x_grey, t.float() / schedule.timesteps)

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1:02d}/{epochs} | MSE loss: {avg_loss:.4f}")

    return loss_history


def chw_to_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.clip(image, 0.0, 1.0)


def save_forward_diffusion_grid(
    grey: np.ndarray,
    rgb: np.ndarray,
    schedule: DiffusionSchedule,
    output_path: Path,
    timesteps: tuple[int, int, int] = (45, 150, 255),
) -> None:
    examples = 3
    y0 = torch.from_numpy(rgb[:examples]).float()

    fig = plt.figure(figsize=(12, 6))
    for i in range(examples):
        ax = fig.add_subplot(examples, 5, i * 5 + 1)
        ax.imshow(grey[i].transpose(1, 2, 0).squeeze(), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("Input (x)")

        ax = fig.add_subplot(examples, 5, i * 5 + 2)
        ax.imshow(chw_to_hwc(rgb[i]))
        ax.axis("off")
        if i == 0:
            ax.set_title("Clean (y0)")

        for j, t_value in enumerate(timesteps):
            t = torch.tensor([t_value], dtype=torch.long)
            yt, _ = schedule.add_noise(y0[i : i + 1], t)
            ax = fig.add_subplot(examples, 5, i * 5 + 3 + j)
            ax.imshow(chw_to_hwc(yt[0]))
            ax.axis("off")
            if i == 0:
                ax.set_title(f"t={t_value}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(loss_history: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history)
    ax.set_title("Diffusion Denoiser Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_one_step_denoising_grid(
    model: Denoiser,
    grey: np.ndarray,
    rgb: np.ndarray,
    schedule: DiffusionSchedule,
    device: torch.device,
    output_path: Path,
    timestep: int = 100,
) -> None:
    model.eval()
    examples = 5
    x_grey = torch.from_numpy(grey[:examples]).float().to(device)
    y0 = torch.from_numpy(rgb[:examples]).float().to(device)
    t = torch.full((examples,), timestep, device=device, dtype=torch.long)

    with torch.no_grad():
        yt, _ = schedule.add_noise(y0, t)
        noise_pred = model(yt, x_grey, t.float() / schedule.timesteps)
        y0_hat = schedule.reconstruct_x0(yt, t, noise_pred)

    fig = plt.figure(figsize=(10, 6))
    for i in range(examples):
        ax = fig.add_subplot(3, examples, i + 1)
        ax.imshow(chw_to_hwc(y0[i]))
        ax.axis("off")
        if i == 0:
            ax.set_title("Clean y0")

        ax = fig.add_subplot(3, examples, i + 1 + examples)
        ax.imshow(chw_to_hwc(yt[i]))
        ax.axis("off")
        if i == 0:
            ax.set_title(f"Noisy yt (t={timestep})")

        ax = fig.add_subplot(3, examples, i + 1 + 2 * examples)
        ax.imshow(chw_to_hwc(y0_hat[i]))
        ax.axis("off")
        if i == 0:
            ax.set_title("Predicted y0_hat")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional diffusion denoiser for CIFAR-10 horse colorization.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory for CIFAR-10 data.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated figures.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true", help="Disable CIFAR-10 download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_grey, train_rgb, test_grey, test_rgb = load_cifar10_horses(
        root=args.data_dir,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        download=not args.no_download,
    )

    schedule = DiffusionSchedule.linear(timesteps=args.timesteps, device=device)
    cpu_schedule = DiffusionSchedule.linear(timesteps=args.timesteps, device="cpu")
    loader = make_loader(train_grey, train_rgb, args.batch_size)

    save_forward_diffusion_grid(
        test_grey,
        test_rgb,
        cpu_schedule,
        args.output_dir / "forward_diffusion.png",
    )

    model = Denoiser(num_filters=32).to(device)
    loss_history = train_denoiser(
        model,
        loader,
        schedule,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    save_loss_curve(loss_history, args.output_dir / "denoiser_training_loss.png")
    save_one_step_denoising_grid(
        model,
        test_grey,
        test_rgb,
        schedule,
        device,
        args.output_dir / "one_step_denoising.png",
    )


if __name__ == "__main__":
    main()
