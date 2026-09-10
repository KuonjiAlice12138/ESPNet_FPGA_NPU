"""Shared geometry contract for compiler, replay, export and evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DeploymentGeometry:
    input_height: int
    input_width: int
    target_scale: int = 8

    def __post_init__(self) -> None:
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input height and width must be positive")
        if self.target_scale <= 0:
            raise ValueError("target scale must be positive")
        if self.input_height % self.target_scale or self.input_width % self.target_scale:
            raise ValueError("input height and width must be divisible by target scale")

    @property
    def logits_height(self) -> int:
        return self.input_height // self.target_scale

    @property
    def logits_width(self) -> int:
        return self.input_width // self.target_scale

    @property
    def input_shape_nhwc(self) -> tuple[int, int, int]:
        return self.input_height, self.input_width, 3

    @property
    def input_bytes(self) -> int:
        return self.input_height * self.input_width * 3

    @property
    def output_mask_bytes(self) -> int:
        return self.input_height * self.input_width

    def lowres_shape(self, class_count: int | None = None) -> tuple[int, ...]:
        shape = (self.logits_height, self.logits_width)
        return shape if class_count is None else shape + (int(class_count),)

    def to_manifest(self) -> dict[str, int]:
        return {
            "input_height": int(self.input_height),
            "input_width": int(self.input_width),
            "logits_height": int(self.logits_height),
            "logits_width": int(self.logits_width),
            "target_scale": int(self.target_scale),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "DeploymentGeometry":
        required = ("input_height", "input_width", "target_scale")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"geometry manifest missing fields: {missing}")
        geometry = cls(
            input_height=int(value["input_height"]),
            input_width=int(value["input_width"]),
            target_scale=int(value["target_scale"]),
        )
        declared_logits = (
            int(value["logits_height"]),
            int(value["logits_width"]),
        )
        if declared_logits != (geometry.logits_height, geometry.logits_width):
            raise ValueError(
                "geometry manifest logits shape disagrees with input shape: "
                f"declared={declared_logits} derived={(geometry.logits_height, geometry.logits_width)}"
            )
        return geometry


DEFAULT_GEOMETRY = DeploymentGeometry(256, 512, 8)
LEGACY_GEOMETRY = DeploymentGeometry(512, 1024, 8)


def geometry_from_manifest(
    manifest: Mapping[str, Any],
    *,
    allow_legacy_fallback: bool = True,
) -> DeploymentGeometry:
    value = manifest.get("geometry")
    if value is None:
        if allow_legacy_fallback:
            return LEGACY_GEOMETRY
        raise ValueError("manifest has no geometry section")
    return DeploymentGeometry.from_manifest(value)
