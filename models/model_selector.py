"""Minimal model selector used by `main.py`."""

from typing import Any, Optional

import torch


def get_model(
    model_name: str,
    modality: str,
    args: Any,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Build a model instance for the cleaned distillation pipeline.

    Supported combinations:
    - modality='rgb', model_name='FactorizePhys'
    - modality='rf', model_name='RF_conv_decoder'
    """
    modality = modality.lower()

    if modality == "rgb":
        if model_name != "FactorizePhys":
            raise ValueError(
                f"Unsupported RGB model '{model_name}' in cleaned project. "
                "Only 'FactorizePhys' is kept."
            )
        from models.rgb.FactorizePhys import FactorizePhys

        frames = getattr(args, "frame_length", 256)
        channels = getattr(args, "channels", 3)
        model = FactorizePhys(frames=frames, in_channels=channels)

    elif modality == "rf":
        if model_name != "RF_conv_decoder":
            raise ValueError(
                f"Unsupported RF model '{model_name}' in cleaned project. "
                "Only 'RF_conv_decoder' is kept."
            )
        from models.rf.RF_conv_decoder import RF_conv_decoder

        channels = getattr(args, "channels", 10)
        model = RF_conv_decoder(channels=channels)
    else:
        raise ValueError(
            f"Unsupported modality '{modality}' in cleaned project. "
            "Only 'rgb' and 'rf' are kept."
        )

    if device is not None:
        model = model.to(device)
    return model


def get_supported_models(modality: str):
    modality = modality.lower()
    if modality == "rgb":
        return ["FactorizePhys"]
    if modality == "rf":
        return ["RF_conv_decoder"]
    return []
