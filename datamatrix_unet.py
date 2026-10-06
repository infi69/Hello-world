from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.maxpool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class StructuralFusionBlock(nn.Module):
    """Fuse decoder features with an external structure prior."""

    def __init__(self, channels: int):
        super().__init__()
        self.structure_proj = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor, structure: Optional[torch.Tensor]) -> torch.Tensor:
        if structure is None:
            return feat

        structure = F.interpolate(structure, size=feat.shape[2:], mode="bilinear", align_corners=False)
        structure = self.structure_proj(structure)
        feat = torch.cat([feat, structure], dim=1)
        return self.feature_fuse(feat)


class LocalGlobalAttentionBlock(nn.Module):
    """Merge local attention and global attention for module repair."""

    def __init__(self, channels: int):
        super().__init__()
        reduced = max(1, channels // 4)

        self.local_attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_attn = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_map = self.local_attn(x)
        global_map = self.global_attn(self.global_pool(x))

        x_local = x * local_map
        x_global = x * global_map

        return self.fuse(torch.cat([x_local, x_global], dim=1))


class RecurrentFeatureReasoningBlock(nn.Module):
    """Iterative feature repair with shared weights and structure feedback."""

    def __init__(self, channels: int, num_steps: int = 3):
        super().__init__()
        self.num_steps = num_steps

        self.initial_fusion = StructuralFusionBlock(channels)
        self.structure_proj = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attn = LocalGlobalAttentionBlock(channels)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.repair_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat: torch.Tensor, structure: Optional[torch.Tensor]) -> torch.Tensor:
        state = self.initial_fusion(feat, structure)

        if structure is None:
            structure = torch.sigmoid(self.repair_head(state))
        else:
            structure = F.interpolate(structure, size=state.shape[2:], mode="bilinear", align_corners=False)

        for _ in range(self.num_steps):
            structure_feat = self.structure_proj(structure)
            guided = state + structure_feat
            attn_feat = self.attn(guided)
            delta = self.refine(attn_feat)
            state = self.act(state + self.gate(attn_feat) * delta)

            repaired_prob = torch.sigmoid(self.repair_head(state))
            structure = 0.5 * structure + 0.5 * repaired_prob

        return state


class DataMatrixUNet(nn.Module):
    """U-Net for DataMatrix module restoration.

    Input: grayscale crop normalized to [0, 1].
    Output: probability map of dark modules.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        reasoning_steps: int = 3,
    ):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)

        self.up1 = Up(base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        self.reasoning = RecurrentFeatureReasoningBlock(base_channels, num_steps=reasoning_steps)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, structure: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.reasoning(x, structure)
        logits = self.outc(x)
        return torch.sigmoid(logits)


def preprocess_datamatrix_crop(crop_bgr: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Normalize a crop to the expected grayscale input for the network."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty crop provided")

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    normalized = clahe.astype(np.float32) / 255.0
    resized = cv2.resize(normalized, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return resized[None, :, :]


def build_structure_prior(gray: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Create a DataMatrix-aware foreground prior.

    Uses:
    - contrast enhancement
    - Otsu foreground extraction
    - weak orientation-agnostic L-finder bias
    """
    if gray is None or gray.size == 0:
        raise ValueError("Empty grayscale image provided")

    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (bw < 128).astype(np.float32)

    h, w = mask.shape
    edge_scale_x = max(1.0, 0.12 * w)
    edge_scale_y = max(1.0, 0.12 * h)

    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)

    left = np.exp(-x / edge_scale_x)[None, :].repeat(h, axis=0)
    right = np.exp(-(w - 1 - x) / edge_scale_x)[None, :].repeat(h, axis=0)
    top = np.exp(-y / edge_scale_y)[:, None].repeat(w, axis=1)
    bottom = np.exp(-(h - 1 - y) / edge_scale_y)[:, None].repeat(w, axis=1)

    l_top_left = np.clip(top + left, 0.0, 1.0)
    l_top_right = np.clip(top + right, 0.0, 1.0)
    l_bottom_left = np.clip(bottom + left, 0.0, 1.0)
    l_bottom_right = np.clip(bottom + right, 0.0, 1.0)

    l_bias = np.maximum.reduce([l_top_left, l_top_right, l_bottom_left, l_bottom_right])
    prior = np.clip(0.8 * mask + 0.2 * l_bias, 0.0, 1.0)

    prior = cv2.resize(prior, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return prior[None, None, :, :].astype(np.float32)


def predict_probability_map(
    model: nn.Module,
    crop_bgr: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return the raw probability map predicted by the model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    input_map = preprocess_datamatrix_crop(crop_bgr)
    structure = build_structure_prior(gray)

    with torch.no_grad():
        x = torch.from_numpy(input_map).float().unsqueeze(0).to(device)
        s = torch.from_numpy(structure).float().to(device)
        pred = model(x, s)[0, 0].detach().cpu().numpy()

    return pred


def predict_module_map(
    model: nn.Module,
    crop_bgr: np.ndarray,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a binary module map inferred by the DataMatrix U-Net."""
    prob = predict_probability_map(model, crop_bgr, device=device)
    return (prob > threshold).astype(np.uint8)


def render_probability_map(prob_map: np.ndarray, module_px: int = 12, quiet_zone: int = 2) -> np.ndarray:
    """Render a binary or probability map as a larger barcode-style bitmap."""
    h, w = prob_map.shape[:2]
    canvas = np.full(
        (
            h * module_px + 2 * quiet_zone * module_px,
            w * module_px + 2 * quiet_zone * module_px,
        ),
        255,
        dtype=np.uint8,
    )

    for y in range(h):
        for x in range(w):
            if prob_map[y, x] > 0.5:
                y0 = (y + quiet_zone) * module_px
                y1 = y0 + module_px
                x0 = (x + quiet_zone) * module_px
                x1 = x0 + module_px
                canvas[y0:y1, x0:x1] = 0

    return canvas


def save_model_checkpoint(model: nn.Module, checkpoint_path: str | Path | None = None) -> Path:
    """Save the current model state to a checkpoint file.

    This creates a real .pt file so the loader in main.py can find it.
    """
    root_dir = Path(__file__).resolve().parent
    if checkpoint_path is None:
        ckpt_dir = root_dir / "weights"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / "datamatrix_unet_best.pt"
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root_dir / checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "config": {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 32,
            "reasoning_steps": 3,
        },
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


def make_synthetic_datamatrix_sample(size: int = 256, grid_size: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic DataMatrix-like crop and corresponding binary target.

    The target is a clean foreground mask with a finder-like border and module noise,
    which is enough to train a lightweight U-Net for initialization.
    """
    canvas = np.full((size, size), 255, dtype=np.uint8)
    grid = np.ones((grid_size, grid_size), dtype=np.uint8)

    finder = 3
    grid[:finder, :finder] = 0
    grid[:finder, -finder:] = 0
    grid[-finder:, :finder] = 0

    # Add a weak alternating-bar structure to simulate a module pattern.
    for r in range(grid_size):
        for c in range(grid_size):
            if r < finder and c < finder:
                continue
            if r < finder and c >= grid_size - finder:
                continue
            if r >= grid_size - finder and c < finder:
                continue
            if (r + c) % 2 == 0:
                grid[r, c] = 0

    # Extra random module corruption to avoid perfectly uniform labels.
    noise = np.random.rand(grid_size, grid_size) < 0.12
    grid[noise] = 0

    # Convert grid to a dense barcode image with quiet zone.
    module_px = max(2, size // (grid_size + 8))
    quiet_zone = 2
    block = np.zeros((grid_size * module_px, grid_size * module_px), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] == 0:
                y0 = r * module_px
                y1 = (r + 1) * module_px
                x0 = c * module_px
                x1 = (c + 1) * module_px
                block[y0:y1, x0:x1] = 0
            else:
                block[r * module_px:(r + 1) * module_px, c * module_px:(c + 1) * module_px] = 255

    # Place in a larger quiet-zone canvas.
    margin = quiet_zone * module_px
    target = np.full((size, size), 255, dtype=np.uint8)
    h, w = block.shape
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    target[y0:y0 + h, x0:x0 + w] = block

    target_mask = (target < 128).astype(np.float32)

    # Create a realistic grayscale barcode image from the binary target.
    image = np.full((size, size), 255, dtype=np.float32)
    image[target_mask > 0.5] = 0.0
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.addWeighted(image, 0.85, np.full_like(image, 200, dtype=np.float32), 0.15, 0.0)
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    return image, target_mask


def train_datamatrix_unet(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    checkpoint_prefix: str = "datamatrix_unet_epoch",
):
    """Train a lightweight DataMatrix U-Net on synthetic barcode-like images.

    This creates a real checkpoint after each epoch and is meant to provide a working
    starting point before moving to a real labeled dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DataMatrixUNet(reasoning_steps=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(12):
            inputs = []
            targets = []
            structures = []

            for _ in range(batch_size):
                image, target = make_synthetic_datamatrix_sample(size=256)
                gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if len(gray.shape) == 3 else gray

                inputs.append(preprocess_datamatrix_crop(image, target_size=256))
                targets.append(target.astype(np.float32))
                structures.append(build_structure_prior(gray, target_size=256))

            x = torch.from_numpy(np.stack(inputs)).float().to(device)
            y = torch.from_numpy(np.stack(targets)).float().unsqueeze(1).to(device)
            s = torch.from_numpy(np.stack(structures)).float().to(device)

            optimizer.zero_grad()
            pred = model(x, s)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        checkpoint_path = save_model_checkpoint(
            model,
            Path("weights") / f"{checkpoint_prefix}_{epoch}.pt",
        )
        print(f"epoch={epoch} loss={epoch_loss / 12:.6f} checkpoint={checkpoint_path}")

    return model


if __name__ == "__main__":
    train_datamatrix_unet(epochs=5, batch_size=4, learning_rate=1e-3)