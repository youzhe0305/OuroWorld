"""Download the model weights into ``checkpoints/`` (the paths of ``configs/*.yaml``).

Usage::

    python scripts/download_weights.py                  # everything (~55 GB)
    python scripts/download_weights.py --only train     # just the 4D optimisation

Groups: ``train`` (LCM-LoRA; SD 1.5 itself is fetched by diffusers on first
use), ``generation`` (VGGT-Omega, CogVideoX-Fun, TrajectoryCrafter, BLIP-2),
``evaluation`` (VBench models, SEA-RAFT, I3D). VGGT-Omega is gated: request
access at https://huggingface.co/facebook/VGGT-Omega, then set ``HF_TOKEN``
(or run ``huggingface-cli login``). Existing files are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"
VBENCH_FILES = {
    "clip_model/ViT-B-32.pt": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "clip_model/ViT-L-14.pt": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "dino_model/dino_vitbase16_pretrain.pth": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth": "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth",
    "ViCLIP/ViClip-InternVid-10M-FLT.pth": "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth",
    "ViCLIP/bpe_simple_vocab_16e6.txt.gz": "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz",
}  # fmt: skip
DINO_REPOSITORY = "https://github.com/facebookresearch/dino.git"
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"


def hub_file(repo: str, filename: str, destination: Path) -> None:
    """One file of a Hugging Face model repository."""
    from huggingface_hub import hf_hub_download

    if destination.is_file():
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(repo, filename, local_dir=destination.parent))
    if downloaded != destination:
        downloaded.replace(destination)


def hub_snapshot(repo: str, destination: Path, patterns: list[str] | None = None) -> None:
    """A Hugging Face model repository (or the files matching ``patterns``)."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=destination, allow_patterns=patterns)
    print(f"ready: {destination}")


def url_file(url: str, destination: Path, sha256: str | None = None) -> None:
    """A plain download, checked against ``sha256`` when given."""
    if destination.is_file() and (sha256 is None or _sha256(destination) == sha256):
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    # Dropbox answers an expired link with an HTML page, not an error.
    if sha256 is not None and _sha256(partial) != sha256:
        partial.unlink()
        raise RuntimeError(f"{url}: checksum mismatch")
    partial.replace(destination)
    print(f"wrote: {destination}")


def train() -> None:
    """LCM-LoRA of the diffusion refinement."""
    hub_file(
        "latent-consistency/lcm-lora-sdv1-5",
        "pytorch_lora_weights.safetensors",
        CHECKPOINTS / "lcm-lora-sdv1-5" / "pytorch_lora_weights.safetensors",
    )


def generation() -> None:
    """Depth lifting, video inpainting and its caption model."""
    hub_file("facebook/VGGT-Omega", "vggt_omega_1b_512.pt", CHECKPOINTS / "vggt_omega_1b_512.pt")
    hub_snapshot("TrajectoryCrafter/TrajectoryCrafter", CHECKPOINTS / "TrajectoryCrafter")
    # TrajectoryCrafter replaces the transformer; only the rest of CogVideoX-Fun is used.
    hub_snapshot(
        "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        CHECKPOINTS / "CogVideoX-Fun-V1.1-5b-InP",
        [
            "model_index.json",
            "configuration.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "vae/*",
        ],
    )
    hub_snapshot(
        "Salesforce/blip2-opt-2.7b",
        CHECKPOINTS / "blip2-opt-2.7b",
        ["*.json", "*.txt", "*.safetensors"],
    )


def evaluation() -> None:
    """VBench's models, SEA-RAFT and I3D."""
    vbench = CHECKPOINTS / "vbench"
    for relative, url in VBENCH_FILES.items():
        url_file(url, vbench / relative)
    dino = vbench / "dino_model" / "facebookresearch_dino_main"
    if not (dino / "hubconf.py").is_file():
        subprocess.run(["git", "clone", "--depth", "1", DINO_REPOSITORY, str(dino)], check=True)
    hub_snapshot("MemorySlices/Tartan-C-T-TSKH-spring540x960-M", CHECKPOINTS / "sea-raft-spring-M")
    url_file(I3D_URL, CHECKPOINTS / "i3d_torchscript.pt", I3D_SHA256)


GROUPS = {"train": train, "generation": generation, "evaluation": evaluation}


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=GROUPS, action="append", help="default: all groups")
    arguments = parser.parse_args()
    for name in arguments.only or GROUPS:
        print(f"== {name}")
        GROUPS[name]()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
