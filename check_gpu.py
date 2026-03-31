from __future__ import annotations

import json

import torch

from src.sdp_pipeline import get_torch_device, get_xgboost_device, gpu_xgboost_ready


def main() -> None:
    payload = {
        "torch_version": torch.__version__,
        "torch_cuda_built_for": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device_count": torch.cuda.device_count(),
        "active_torch_device": str(get_torch_device()),
        "active_xgboost_device": get_xgboost_device(),
        "gpu_xgboost_ready": gpu_xgboost_ready(),
    }
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
