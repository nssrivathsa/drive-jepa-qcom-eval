import os
import argparse
from huggingface_hub import hf_hub_download

CHECKPOINTS = {
    "resnet34": "drive_jepa_perception_based_agent_resnet34.ckpt",
    "resnet34_v2": "drive_jepa_perception_based_agent_resnet34_v2.ckpt",
    "vitl_perception_free": "drive_jepa_perception_free_agent_vitl.ckpt",
    "vitl_perception_based": "drive_jepa_perception_based_agent_vitl.ckpt",
}


def download_checkpoint(model_key: str = "resnet34", output_dir: str = "weights"):
    """
    Downloads Drive-JEPA pre-trained checkpoint from Hugging Face (repo: LinhanWang/Drive-JEPA).
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = CHECKPOINTS.get(model_key, CHECKPOINTS["resnet34"])

    dest_path = os.path.join(output_dir, filename)
    if os.path.exists(dest_path):
        print(f"[Drive-JEPA] Checkpoint already exists at: {dest_path}")
        return dest_path

    print(f"[Drive-JEPA] Downloading {filename} from Hugging Face (repo: LinhanWang/Drive-JEPA)...")
    downloaded_path = hf_hub_download(
        repo_id="LinhanWang/Drive-JEPA",
        repo_type="dataset",
        filename=filename,
        local_dir=output_dir,
    )
    print(f"[Drive-JEPA] Download complete! Saved to: {downloaded_path}")
    return downloaded_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Drive-JEPA Checkpoints from Hugging Face")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
        choices=list(CHECKPOINTS.keys()),
        help="Model checkpoint to download (default: resnet34)",
    )
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save weights")
    args = parser.parse_args()

    download_checkpoint(args.model, args.output_dir)
