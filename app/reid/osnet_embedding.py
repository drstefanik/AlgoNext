from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

logger = logging.getLogger(__name__)

OSNET_MODEL_NAME = "osnet_x0_25"
OSNET_WEIGHT_FILENAME = (
    "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
    "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
OSNET_DESCRIPTOR_VERSION = "osnet-x0.25-msmt17+hsv-torso-v2"
DEFAULT_OSNET_MODEL_PATH = f"/opt/algonext-models/{OSNET_WEIGHT_FILENAME}"

_INFERENCE_LOCK = threading.Lock()
_LOAD_ERROR_REPORTED = False


def configured_model_path() -> Path:
    return Path(
        os.environ.get("PLAYER_REID_OSNET_MODEL_PATH", DEFAULT_OSNET_MODEL_PATH)
    )


def _load_osnet_source_module() -> Any:
    distribution = importlib.metadata.distribution("torchreid")
    module_path = distribution.locate_file("torchreid/reid/models/osnet.py")
    spec = importlib.util.spec_from_file_location(
        "_algonext_torchreid_osnet",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the packaged OSNet architecture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("OSNet checkpoint must be a mapping")
    value = checkpoint.get("state_dict")
    return value if isinstance(value, Mapping) else checkpoint


@lru_cache(maxsize=2)
def _load_model(model_path_value: str) -> Any:
    import torch

    model_path = Path(model_path_value)
    if not model_path.is_file():
        raise FileNotFoundError(f"OSNet weights not found: {model_path}")
    osnet = _load_osnet_source_module()
    model = osnet.osnet_x0_25(
        num_classes=1,
        pretrained=False,
        loss="softmax",
        use_gpu=False,
    )
    checkpoint = torch.load(
        str(model_path),
        map_location="cpu",
        weights_only=True,
    )
    source = _state_dict(checkpoint)
    target = model.state_dict()
    compatible: dict[str, Any] = {}
    for raw_key, value in source.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[7:]
        if key in target and getattr(value, "shape", None) == target[key].shape:
            compatible[key] = value
    if len(compatible) < 20:
        raise RuntimeError(
            "OSNet checkpoint did not contain enough compatible feature layers"
        )
    target.update(compatible)
    model.load_state_dict(target)
    model.eval()
    model.to("cpu")
    return model


def extract_osnet_embedding(crop: np.ndarray) -> tuple[float, ...] | None:
    """Return a normalized, person-ReID-specific OSNet embedding.

    Loading is lazy and fail-closed inside this provider. The caller decides
    whether an unavailable learned model may fall back to the legacy descriptor.
    """

    global _LOAD_ERROR_REPORTED
    if crop is None or crop.ndim != 3 or crop.shape[0] < 16 or crop.shape[1] < 8:
        return None
    try:
        import torch

        model = _load_model(str(configured_model_path()))
        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).float()
        tensor = tensor.div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0)
        with _INFERENCE_LOCK, torch.no_grad():
            vector = model(tensor).reshape(-1)
            norm = torch.linalg.vector_norm(vector)
            if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
                return None
            vector = vector / norm
        return tuple(float(item) for item in vector.cpu().tolist())
    except Exception:
        if not _LOAD_ERROR_REPORTED:
            logger.exception(
                "OSNet ReID embedding unavailable; descriptor policy will "
                "decide whether to use the conservative HSV fallback"
            )
            _LOAD_ERROR_REPORTED = True
        return None
