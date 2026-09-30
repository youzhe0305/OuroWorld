"""Compatibility shims between pinned third-party versions."""

from __future__ import annotations

from typing import Any


def allow_sdpa_enable_gqa() -> None:
    """Accept ``enable_gqa=False`` in ``scaled_dot_product_attention`` on PyTorch < 2.5.

    diffusers 0.35 always passes the keyword, which PyTorch added in 2.5. The
    shim drops it when grouped-query attention is off, so the computation is
    unchanged; it refuses the call otherwise.
    """
    import torch
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) >= Version("2.5.0"):
        return
    original = torch.nn.functional.scaled_dot_product_attention
    if getattr(original, "_accepts_enable_gqa", False):
        return

    def scaled_dot_product_attention(*args: Any, **kwargs: Any) -> Any:
        if kwargs.pop("enable_gqa", False):
            raise RuntimeError("grouped-query attention needs PyTorch >= 2.5")
        return original(*args, **kwargs)

    scaled_dot_product_attention._accepts_enable_gqa = True  # type: ignore[attr-defined]
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
