import argparse
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
from huggingface_hub import snapshot_download
from muq import MuQ, MuQMuLan

from domain import config
from domain.enums import EmbeddingModel
from inference.models import clap_model, mert_model, muq_model, muq_mulan_model

CLAP_DOWNLOAD_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
)


def get_target_path(model_name: str, models_dir: Path) -> Path:
    """Returns the filesystem target path for a given model under models_dir."""
    if model_name == EmbeddingModel.CLAP.value:
        return models_dir / os.path.basename(clap_model.MODEL_CHECKPOINT_PATH)
    elif model_name == EmbeddingModel.MERT.value:
        return models_dir / os.path.basename(mert_model.LOCAL_MODEL_PATH)
    elif model_name == EmbeddingModel.MUQ.value:
        return models_dir / os.path.basename(muq_model.MODEL_PATH)
    elif model_name == EmbeddingModel.MUQ_MULAN.value:
        return models_dir / os.path.basename(muq_mulan_model.MODEL_PATH)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def download_clap(target_path: Path) -> None:
    """Streams CLAP checkpoint to a temporary file, renamed to target on success."""
    tmp_path = Path(f"{target_path}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with httpx.stream("GET", CLAP_DOWNLOAD_URL, follow_redirects=True) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def download_mert(target_path: Path) -> None:
    """Downloads MERT model from Hugging Face Hub snapshot."""
    snapshot_download(
        repo_id=mert_model.MODEL_NAME,
        local_dir=str(target_path),
        repo_type="model",
        local_dir_use_symlinks=False,
    )


def download_muq(target_path: Path) -> None:
    """Downloads MuQ model into target cache directory."""
    MuQ.from_pretrained(muq_model.REPO_ID, cache_dir=str(target_path))


def download_muq_mulan(target_path: Path) -> None:
    """Downloads MuQ-MuLan model into target cache directory."""
    MuQMuLan.from_pretrained(muq_mulan_model.REPO_ID, cache_dir=str(target_path))


DOWNLOADERS: dict[str, Callable[[Path], None]] = {
    EmbeddingModel.CLAP.value: download_clap,
    EmbeddingModel.MERT.value: download_mert,
    EmbeddingModel.MUQ.value: download_muq,
    EmbeddingModel.MUQ_MULAN.value: download_muq_mulan,
}


def resolve_models(models_arg: str | None = None) -> list[str]:
    """Resolves the models to download based on config.ANALYZE_MODELS or CLI --models.

    muq_mulan is always added.
    acoustic_features is excluded as it has no model weights to download.
    """
    if models_arg is not None:
        analysis_part = [m.strip() for m in models_arg.split(",") if m.strip()]
        valid_models = {m.value for m in EmbeddingModel}
        for m in analysis_part:
            if m not in valid_models:
                raise ValueError(
                    f"Unknown model '{m}'. Valid choices are: {', '.join(sorted(valid_models))}"
                )
    else:
        analysis_part = list(config.ANALYZE_MODELS)

    target_set = (set(analysis_part) | {EmbeddingModel.MUQ_MULAN.value}) - {
        EmbeddingModel.ACOUSTIC_FEATURES.value
    }
    preferred_order = [
        EmbeddingModel.MUQ.value,
        EmbeddingModel.MUQ_MULAN.value,
        EmbeddingModel.CLAP.value,
        EmbeddingModel.MERT.value,
    ]
    return [m for m in preferred_order if m in target_set] + sorted(
        target_set - set(preferred_order)
    )


def download_model(model_name: str, models_dir: Path) -> str:
    """Downloads or adopts a single model.

    Returns: 'skip', 'adopt existing', or 'downloaded'.
    """
    target_path = get_target_path(model_name, models_dir)
    marker_path = Path(f"{target_path}.complete")

    if marker_path.exists():
        print(f"{model_name}: skip")
        return "skip"

    if target_path.exists():
        if target_path.is_dir():
            if any(target_path.iterdir()):
                marker_path.touch()
                print(f"{model_name}: adopt existing")
                return "adopt existing"
            else:
                target_path.rmdir()
        else:
            target_path.unlink()

    tmp_path = Path(f"{target_path}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    downloader = DOWNLOADERS[model_name]
    try:
        downloader(target_path)
    except Exception as e:
        if marker_path.exists():
            marker_path.unlink()
        if tmp_path.exists():
            tmp_path.unlink()
        if target_path.is_dir():
            shutil.rmtree(target_path, ignore_errors=True)
        elif target_path.is_file():
            target_path.unlink(missing_ok=True)
        raise e

    marker_path.touch()
    print(f"{model_name}: downloaded")
    return "downloaded"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model weights for inference.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to download (default: LAINBOW_ANALYZE_MODELS + muq_mulan).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./models",
        help="Directory to store downloaded models (default: ./models).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models_dir = Path(args.dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        models_to_download = resolve_models(args.models)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    failed = False
    for model_name in models_to_download:
        try:
            download_model(model_name, models_dir)
        except Exception as e:
            print(f"Error downloading {model_name}: {e}", file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
