"""Shared geometry contract for compiler, replay, export and evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from numbers import Integral
from typing import Any, Mapping


@dataclass(frozen=True)
class DeploymentGeometry:
    input_height: int
    input_width: int
    target_scale: int = 8
    input_channels: int = 3

    def __post_init__(self) -> None:
        if self.input_channels not in (1, 3):
            raise ValueError("input channels must be 1 (gray) or 3 (RGB)")
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
        return self.input_height, self.input_width, self.input_channels

    @property
    def input_bytes(self) -> int:
        return self.input_height * self.input_width * self.input_channels

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
            "input_channels": int(self.input_channels),
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
            input_channels=int(value.get("input_channels", 3)),
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


# Profile and source IDs are a runtime ABI, not a list of trained checkpoints.
RUNTIME_ABI_ID = 0x11160001
RUNTIME_CAPABILITY = 0x7FFF  # profile[7:0], native/lite-C1/lite-C3/ROI256/ROI512/class2/class20
RUNTIME_LAYOUT_VERSION = "p7-unified-layout-v1"


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: int
    name: str
    geometry: DeploymentGeometry


RUNTIME_PROFILES = tuple(
    RuntimeProfile(index, f"h{h}w{w}_{'gray' if c == 1 else 'rgb'}",
                   DeploymentGeometry(h, w, 8, c))
    for index, (h, w, c) in enumerate(
        (h, w, c) for h, w in ((240, 240), (256, 512), (512, 512), (512, 1024))
        for c in (1, 3)
    )
)
LITE_SOURCES = {
    1: (256, 512, 1), 2: (512, 1024, 1), 3: (512, 512, 1),
    4: (256, 512, 3), 5: (512, 1024, 3), 6: (512, 512, 3),
}


def _runtime_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def runtime_profile(profile_id: int) -> RuntimeProfile:
    profile_id = _runtime_integer(profile_id, "profile_id")
    if not 0 <= profile_id < len(RUNTIME_PROFILES):
        raise ValueError(f"unsupported runtime profile_id: {profile_id}")
    return RUNTIME_PROFILES[profile_id]


def profile_for_geometry(geometry: DeploymentGeometry) -> RuntimeProfile:
    for profile in RUNTIME_PROFILES:
        if (profile.geometry.input_shape_nhwc == geometry.input_shape_nhwc
                and geometry.target_scale == 8):
            return profile
    raise ValueError(f"unsupported runtime encoder geometry: {geometry.to_manifest()}")


@dataclass(frozen=True)
class RuntimeSelection:
    profile: RuntimeProfile
    classes: int
    src_mode: int
    mask_mode: int
    source_shape: tuple[int, int, int]
    roi: tuple[int, int, int, int]  # top, left, height, width in the source image
    mask_shape: tuple[int, int]

    @property
    def input_bytes(self) -> int:
        h, w, c = self.source_shape
        return h * w * c

    @property
    def mask_bytes(self) -> int:
        return self.mask_shape[0] * self.mask_shape[1]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "runtime_abi_id": RUNTIME_ABI_ID,
            "layout_version": RUNTIME_LAYOUT_VERSION,
            "model_profile_id": self.profile.profile_id,
            "model_profile_name": self.profile.name,
            "geometry": self.profile.geometry.to_manifest(),
            "class_count": self.classes,
            "src_mode": self.src_mode, "mask_mode": self.mask_mode,
            "source_shape_nhwc": list(self.source_shape),
            "roi_yxhw": list(self.roi), "mask_shape_hw": list(self.mask_shape),
            "input_format": "NHWC_INT8_PREQUANTIZED",
            "input_resize": "none" if self.src_mode == 0 else "center_square_half_pixel_nearest",
            "output_resize": "half_pixel_linear_logits_then_argmax",
        }


def runtime_selection(profile_id: int, classes: int, *, src_mode: int = 0,
                      mask_mode: int = 0) -> RuntimeSelection:
    profile = runtime_profile(profile_id)
    classes = _runtime_integer(classes, "classes")
    src_mode = _runtime_integer(src_mode, "src_mode")
    mask_mode = _runtime_integer(mask_mode, "mask_mode")
    if classes not in (2, 20) or mask_mode not in (0, 1):
        raise ValueError("runtime supports classes=2/20 and mask_mode=0/1 only")
    encoder = profile.geometry
    if src_mode == 0:
        if mask_mode != 0:
            raise ValueError("native inference requires encoder-resolution mask_mode=0")
        shape = encoder.input_shape_nhwc
        roi = (0, 0, shape[0], shape[1])
        mask = shape[:2]
    else:
        if src_mode not in LITE_SOURCES:
            raise ValueError(f"unsupported src_mode: {src_mode}")
        shape = LITE_SOURCES[src_mode]
        if (encoder.input_height, encoder.input_width, encoder.input_channels) != (240, 240, shape[2]):
            raise ValueError("lite source requires a same-channel 240x240 encoder")
        side = min(shape[:2])
        roi = ((shape[0]-side)//2, (shape[1]-side)//2, side, side)
        mask = (side, side) if mask_mode else (240, 240)
    return RuntimeSelection(profile, classes, src_mode, mask_mode, shape, roi, mask)


def runtime_selection_from_manifest(manifest: Mapping[str, Any]) -> RuntimeSelection:
    value=manifest.get('runtime')
    if not isinstance(value,Mapping):
        raise ValueError('a unified artifact must contain its runtime contract')
    try:
        selected=runtime_selection(value['model_profile_id'],value['class_count'],
                                   src_mode=value['src_mode'],mask_mode=value['mask_mode'])
    except (KeyError,TypeError) as exc:
        raise ValueError('incomplete runtime selection') from exc
    for key,expected in selected.to_manifest().items():
        if value.get(key)!=expected:
            raise ValueError(f'runtime manifest disagrees with canonical {key}')
    return selected


def select_runtime_input(source, selection: RuntimeSelection):
    import numpy as np
    array=np.asarray(source)
    if array.dtype!=np.int8 or array.shape!=selection.source_shape:
        raise ValueError('source must be same-model, prequantized NHWC/HWC signed INT8')
    if selection.src_mode==0:
        return np.ascontiguousarray(array)
    top,left,height,width=selection.roi
    y=np.asarray(nearest_half_pixel_indices(height,240))+top
    x=np.asarray(nearest_half_pixel_indices(width,240))+left
    return np.ascontiguousarray(array[y[:,None],x[None,:],:])


def runtime_mask_from_logits(logits, selection: RuntimeSelection):
    import numpy as np
    g=selection.profile.geometry
    array=np.asarray(logits)
    shape=(g.logits_height,g.logits_width,selection.classes)
    if array.dtype!=np.int8 or (array.shape!=shape and array.shape!=(int(np.prod(shape)),)):
        raise ValueError('logits must be HWC signed INT8 with the declared class count')
    array=array.reshape(shape).astype(np.int64)
    maps=linear_half_pixel_axis(g.logits_width,selection.mask_shape[1])
    x0=np.asarray([v[0] for v in maps]); x1=np.asarray([v[1] for v in maps])
    fx=np.asarray([v[2] for v in maps],dtype=np.int64)[:,None]
    dx=np.asarray([v[3] for v in maps],dtype=np.int64)[:,None]
    mask=np.empty(selection.mask_shape,dtype=np.uint8)
    for y,(y0,y1,fy,dy) in enumerate(linear_half_pixel_axis(g.logits_height,selection.mask_shape[0])):
        top=array[y0,x0]*(dx-fx)+array[y0,x1]*fx
        bottom=array[y1,x0]*(dx-fx)+array[y1,x1]*fx
        mask[y]=np.argmax(top*(dy-fy)+bottom*fy,axis=-1).astype(np.uint8)
    return mask


def runtime_target_view(target, selection: RuntimeSelection, *, center_square_source: bool = False):
    """Original-label image -> declared source -> center ROI -> declared mask grid."""
    import numpy as np
    array=np.asarray(target)
    if array.ndim!=2 or not np.issubdtype(array.dtype,np.integer):
        raise ValueError('target must be a two-dimensional integer label image')
    if center_square_source:
        h,w=array.shape; side=min(h,w)
        array=array[(h-side)//2:(h-side)//2+side,(w-side)//2:(w-side)//2+side]
    sh,sw,_=selection.source_shape
    ys=np.asarray(nearest_half_pixel_indices(array.shape[0],sh))
    xs=np.asarray(nearest_half_pixel_indices(array.shape[1],sw))
    array=array[ys[:,None],xs[None,:]]
    top,left,h,w=selection.roi
    array=array[top:top+h,left:left+w]
    ys=np.asarray(nearest_half_pixel_indices(h,selection.mask_shape[0]))
    xs=np.asarray(nearest_half_pixel_indices(w,selection.mask_shape[1]))
    return np.ascontiguousarray(array[ys[:,None],xs[None,:]])


def nearest_half_pixel_indices(source: int, output: int) -> tuple[int, ...]:
    source = _runtime_integer(source, "source")
    output = _runtime_integer(output, "output")
    if source <= 0 or output <= 0:
        raise ValueError("sampling dimensions must be positive")
    return tuple(min(((2*d+1)*source)//(2*output), source-1) for d in range(output))


def runtime_io_fields(selection: RuntimeSelection) -> dict[str, int]:
    """Finite IO/DDA constants shared by HLS and software, without pixel-loop division."""
    g = selection.profile.geometry
    side = selection.roi[2]
    mask_h, mask_w = selection.mask_shape
    divisor = gcd(g.logits_height, mask_h)
    if g.logits_height * mask_w != g.logits_width * mask_h:
        raise ValueError("the finite mask modes require equal axis scale, not a stretched ROI")
    return {
        'model_profile_id': selection.profile.profile_id,
        'src_mode': selection.src_mode, 'mask_mode': selection.mask_mode,
        'source_h': selection.source_shape[0], 'source_w': selection.source_shape[1],
        'source_c': selection.source_shape[2], 'input_bytes': selection.input_bytes,
        'roi_top': selection.roi[0], 'roi_left': selection.roi[1],
        'roi_h': selection.roi[2], 'roi_w': selection.roi[3],
        'mask_h': mask_h, 'mask_w': mask_w, 'mask_bytes': selection.mask_bytes,
        'nearest_index': side//480 if selection.src_mode else 0,
        'nearest_error': side%480 if selection.src_mode else 0,
        'nearest_base_step': (2*side)//480 if selection.src_mode else 0,
        'nearest_error_step': (2*side)%480 if selection.src_mode else 0,
        'axis_denominator': 2*mask_h//divisor,
        'axis_step': 2*g.logits_height//divisor,
        'axis_origin': (g.logits_height-mask_h)//divisor,
    }


def linear_half_pixel_axis(source: int, output: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return exact (lower, upper, fraction, denominator), with clamped edges."""
    source = _runtime_integer(source, "source")
    output = _runtime_integer(output, "output")
    if source <= 0 or output <= 0:
        raise ValueError("interpolation dimensions must be positive")
    divisor = gcd(source, output)
    denominator = 2*output//divisor
    samples = []
    for d in range(output):
        numerator = ((2*d+1)*source-output)//divisor
        numerator = min(max(numerator, 0), (source-1)*denominator)
        lower, fraction = divmod(numerator, denominator)
        upper = min(lower+1, source-1) if fraction else lower
        samples.append((lower, upper, fraction, denominator))
    return tuple(samples)
