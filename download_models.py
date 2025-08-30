import os
import sys

import httpx
from huggingface_hub import snapshot_download
from muq import MuQ, MuQMuLan

MODELS_DIR = "./models"
MERT_MODEL_PATH = os.path.join(MODELS_DIR, "MERT-v1-330M")
CLAP_MODEL_PATH = os.path.join(MODELS_DIR, "music_audioset_epoch_15_esc_90.14.pt")
MUQ_MODEL_PATH = os.path.join(MODELS_DIR, "MuQ-large-msd-iter")
MUQ_MULAN_MODEL_PATH = os.path.join(MODELS_DIR, "MuQ-MuLan-large")

MERT_REPO_ID = "m-a-p/MERT-v1-330M"
CLAP_DOWNLOAD_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
)
MULAN_REPO_ID = "OpenMuQ/MuQ-MuLan-large"
MUQ_REPO_ID = "OpenMuQ/MuQ-large-msd-iter"


def download_file(url: str, dest_path: str) -> None:
    """Downloads a file from a URL to a destination path with progress."""
    print(f"Downloading {os.path.basename(dest_path)}...")
    try:
        with httpx.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    except httpx.RequestError as e:
        print(f"Error downloading file: {e}", file=sys.stderr)
        # Clean up partially downloaded file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)


def download_mert_model() -> None:
    """Downloads the MERT model from Hugging Face Hub."""
    if os.path.exists(MERT_MODEL_PATH):
        print(f"MERT model already exists at {MERT_MODEL_PATH}. Skipping.")
        return

    print("MERT model not found. Downloading from Hugging Face Hub...")
    try:
        snapshot_download(
            repo_id=MERT_REPO_ID,
            local_dir=MERT_MODEL_PATH,
            repo_type="model",
            local_dir_use_symlinks=False,
        )
        print("MERT model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading MERT model: {e}", file=sys.stderr)
        sys.exit(1)


def download_clap_model() -> None:
    """Downloads the CLAP model checkpoint."""
    if os.path.exists(CLAP_MODEL_PATH):
        print(f"CLAP model already exists at {CLAP_MODEL_PATH}. Skipping.")
        return

    print("CLAP model not found. Downloading...")
    download_file(CLAP_DOWNLOAD_URL, CLAP_MODEL_PATH)


def download_muq_and_muq_mulan_models() -> None:
    if os.path.exists(MUQ_MULAN_MODEL_PATH):
        print(f"MuQ model already exists at {MUQ_MULAN_MODEL_PATH}. Skipping.")
    else:
        MuQMuLan.from_pretrained(MULAN_REPO_ID, cache_dir=MUQ_MULAN_MODEL_PATH)
    if os.path.exists(MUQ_MODEL_PATH):
        print(f"MuQ model already exists at {MUQ_MODEL_PATH}. Skipping.")
    else:
        MuQ.from_pretrained(MUQ_REPO_ID, cache_dir=MUQ_MODEL_PATH)


def main() -> None:
    """Main function to download all necessary models."""
    print("--- Starting Model Download Process ---")
    os.makedirs(MODELS_DIR, exist_ok=True)

    download_mert_model()
    print("-" * 20)
    download_clap_model()
    print("-" * 20)
    download_muq_and_muq_mulan_models()
    print("-" * 20)

    print("\nAll models are downloaded and ready.")
    print("-------------------------------------")


if __name__ == "__main__":
    main()
