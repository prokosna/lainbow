"""Utility script to run model inference using the same logic as batch workers."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def _configure_sys_path() -> None:
    """Ensure the project root (containing ``src``) is on sys.path."""
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if not src_path.exists():
        raise RuntimeError(f"src directory not found at {src_path}")
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_configure_sys_path()

from domain import config as domain_config  # noqa: E402
from domain.schemas import EmbeddingModel  # noqa: E402  (requires sys.path update)
from inference.core.model_manager import ModelManager  # noqa: E402
from worker import clap, mert, muq, muq_mulan  # noqa: E402


MODEL_TO_LOADER = {
    EmbeddingModel.CLAP: clap.get_audio_embeddings_batch,
    EmbeddingModel.MERT: mert.get_audio_embeddings_batch,
    EmbeddingModel.MUQ: muq.get_audio_embeddings_batch,
    EmbeddingModel.MUQ_MULAN: muq_mulan.get_audio_embeddings_batch,
}


_MODEL_MANAGER: ModelManager | None = None


def _initialize_model_manager() -> ModelManager:
    global _MODEL_MANAGER
    if _MODEL_MANAGER is not None:
        return _MODEL_MANAGER

    mode = domain_config.INFERENCE_MODEL_MANAGER_MODE or "single"
    _MODEL_MANAGER = ModelManager(mode=mode)  # type: ignore[arg-type]

    def _local_run_inference(model_name: str, audio_data: np.ndarray) -> np.ndarray:
        assert _MODEL_MANAGER is not None
        return _MODEL_MANAGER.run_inference(model_name, audio_data)

    for module in (clap, mert, muq, muq_mulan):
        setattr(module, "run_inference", _local_run_inference)

    return _MODEL_MANAGER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference locally using the same pipeline as the batch workers, and output the "
            "resulting embedding in JSON for comparison with the production inference API."
        )
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name (e.g. clap, mert, muq, muq_mulan).",
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file to embed.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_embedding(model: EmbeddingModel, audio_path: Path) -> np.ndarray:
    _initialize_model_manager()
    loader = MODEL_TO_LOADER.get(model)
    if loader is None:
        raise ValueError(f"Loader for model '{model.value}' is not configured.")

    embeddings = loader([audio_path])
    path_key = str(audio_path)

    if path_key not in embeddings:
        raise RuntimeError(
            f"Failed to compute embedding. No result returned for '{path_key}'."
        )

    return embeddings[path_key]


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        model = EmbeddingModel(args.model_name)
    except ValueError:
        valid_models = ", ".join(m.value for m in EmbeddingModel)
        logging.error("Unknown model '%s'. Valid options: %s", args.model_name, valid_models)
        return 1

    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.exists():
        logging.error("Audio file not found: %s", audio_path)
        return 1

    try:
        embedding = get_embedding(model, audio_path)
    except Exception:
        logging.exception("Failed to generate embedding.")
        return 1

    output = {
        "model": model.value,
        "audio_path": str(audio_path),
        "embedding": embedding.tolist(),
        "dimension": int(embedding.shape[0]) if embedding.ndim == 1 else list(embedding.shape),
    }

    separators = (", ", ": ") if args.pretty else (",", ":")
    json.dump(
        output,
        sys.stdout,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        separators=separators,
    )
    if args.pretty:
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

