"""Small instruction-conditioned grounder. Every learned parameter starts random."""
from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

PREPROCESSING = "rgb-exif-transpose-bilinear-square-minus1-plus1-v1"
ARCHITECTURE = "cnn-gru-spatial-cross-attention-v1"


class Tokenizer:
    def __init__(self, vocabulary: list[str], max_tokens: int = 32):
        if vocabulary[:2] != ["<pad>", "<unk>"] or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Invalid tokenizer vocabulary")
        self.vocabulary = vocabulary
        self.lookup = {word: i for i, word in enumerate(vocabulary)}
        self.max_tokens = int(max_tokens)

    @staticmethod
    def words(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    @classmethod
    def build(cls, instructions: list[str], max_tokens: int = 32) -> "Tokenizer":
        counts = Counter(word for text in instructions for word in cls.words(text))
        return cls(["<pad>", "<unk>"] + sorted(counts), max_tokens)

    def encode(self, text: str) -> tuple[list[int], int]:
        ids = [self.lookup.get(word, 1) for word in self.words(text)[:self.max_tokens]] or [1]
        return ids + [0] * (self.max_tokens - len(ids)), len(ids)

    def to_dict(self) -> dict:
        return {"vocabulary": self.vocabulary, "max_tokens": self.max_tokens}

    @classmethod
    def from_dict(cls, value: dict) -> "Tokenizer":
        return cls(value["vocabulary"], value["max_tokens"])


def preprocess_image(image: Image.Image | str | Path, image_size: int) -> torch.Tensor:
    """Resize the complete image, without cropping/padding; normalized boxes stay unchanged."""
    if not isinstance(image, Image.Image):
        with Image.open(image) as opened:
            return preprocess_image(opened, image_size)
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
    values = np.array(rgb, dtype=np.float32, copy=True) / 127.5 - 1.0
    return torch.from_numpy(values).permute(2, 0, 1).contiguous()


class Grounder(nn.Module):
    def __init__(self, vocab_size: int, class_count: int, config: dict):
        super().__init__()
        width, dim = config["width"], config["text_dim"]
        layers: list[nn.Module] = []
        channels = 3
        for out_channels in (width, width * 2, width * 2, dim):
            layers.extend([nn.Conv2d(channels, out_channels, 3, stride=2, padding=1),
                           nn.GroupNorm(1, out_channels), nn.GELU()])
            channels = out_channels
        self.image_encoder = nn.Sequential(*layers)
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.text_encoder = nn.GRU(dim, dim, batch_first=True)
        self.position = nn.Linear(6, dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0.0, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.box_head = nn.Linear(dim, 4)
        self.class_head = nn.Linear(dim, class_count)
        self.presence_head = nn.Linear(dim, 1)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> dict:
        visual = self.image_encoder(images)
        batch, channels, height, width = visual.shape
        y, x = torch.meshgrid(torch.linspace(0, 1, height, device=images.device),
                              torch.linspace(0, 1, width, device=images.device), indexing="ij")
        positional = torch.stack((x, y, x.square(), y.square(),
                                  torch.sin(x * torch.pi), torch.sin(y * torch.pi)), dim=-1)
        spatial = visual.flatten(2).transpose(1, 2) + self.position(positional.reshape(-1, 6))
        text_outputs, _ = self.text_encoder(self.embedding(tokens))
        text = text_outputs[torch.arange(batch, device=images.device), lengths - 1]
        attended, _ = self.attention(text.unsqueeze(1), spatial, spatial, need_weights=False)
        fused = self.fusion(torch.cat((attended[:, 0], text), dim=-1))
        # x1/y1 are lower corners; remaining fractions stay inside the unit square.
        fractions = self.box_head(fused).float().sigmoid()
        epsilon = 1e-4
        lower = fractions[:, :2] * (1 - 2 * epsilon)
        upper = lower + epsilon + (1 - lower - epsilon) * fractions[:, 2:]
        return {"bbox": torch.cat((lower, upper), dim=-1),
                "class_logits": self.class_head(fused),
                "presence_logits": self.presence_head(fused).squeeze(-1)}


def box_iou_giou(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    top_left = torch.maximum(predicted[:, :2], target[:, :2])
    bottom_right = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    p_area = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(dim=-1)
    t_area = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(dim=-1)
    union = (p_area + t_area - intersection).clamp(min=1e-8)
    enclosing = (torch.maximum(predicted[:, 2:], target[:, 2:]) -
                 torch.minimum(predicted[:, :2], target[:, :2])).clamp(min=0).prod(dim=-1).clamp(min=1e-8)
    iou = intersection / union
    return iou, iou - (enclosing - union) / enclosing


def grounding_loss(output: dict, batch: dict) -> dict[str, torch.Tensor]:
    positive = batch["present"].bool()
    presence = F.binary_cross_entropy_with_logits(output["presence_logits"].float(), batch["present"].float())
    if positive.any():
        boxes, targets = output["bbox"][positive].float(), batch["boxes"][positive].float()
        l1 = F.l1_loss(boxes, targets)
        giou = (1 - box_iou_giou(boxes, targets)[1]).mean()
        classification = F.cross_entropy(output["class_logits"][positive].float(), batch["classes"][positive])
    else:
        # Keep all heads in the autograd graph, while producing exactly zero masked gradients.
        l1 = output["bbox"].sum() * 0.0
        giou = l1
        classification = output["class_logits"].sum() * 0.0
    return {"total": 5 * l1 + 2 * giou + classification + presence,
            "l1": l1, "giou": giou, "classification": classification, "presence": presence}


def make_batch(records: list[dict], tokenizer: Tokenizer, config: dict, root: Path, device: torch.device) -> dict:
    encoded = [tokenizer.encode(record["instruction"]) for record in records]
    return {
        "images": torch.stack([preprocess_image(root / row["image_path"], config["image_size"]) for row in records]).to(device),
        "tokens": torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device),
        "lengths": torch.tensor([item[1] for item in encoded], dtype=torch.long, device=device),
        "present": torch.tensor([row["target_present"] for row in records], dtype=torch.float32, device=device),
        "boxes": torch.tensor([row["bbox"] if row["target_present"] else [0, 0, 0, 0] for row in records], dtype=torch.float32, device=device),
        "classes": torch.tensor([row["class_id"] if row["target_present"] else 0 for row in records], dtype=torch.long, device=device),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + epoch * 104729)
    return torch.randperm(count, generator=generator).tolist()
