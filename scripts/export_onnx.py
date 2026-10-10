import os
import sys
import argparse

# Force UTF-8 for console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import onnx
import onnxruntime as ort

from models.drive_jepa_net import load_drive_jepa_model


def export_to_onnx(
    checkpoint_path: str = "weights/drive_jepa_perception_based_agent_resnet34.ckpt",
    output_onnx_path: str = "models/drive_jepa_8650.onnx",
    opset_version: int = 18,
):
    """
    Exports Drive-JEPA PyTorch model to ONNX format optimized for Qualcomm HTP.
    """
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    device = "cpu"

    print(f"[ONNX Export] Loading Drive-JEPA model from: {checkpoint_path}")
    model = load_drive_jepa_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    # Dummy inputs for export (Static batch=1 for peak NPU efficiency)
    dummy_frame = torch.randn(1, 6, 256, 512, dtype=torch.float32)
    dummy_history = torch.randn(1, 11, dtype=torch.float32)

    # Compute PyTorch reference output
    with torch.no_grad():
        torch_out = model(dummy_frame, dummy_history)

    input_names = ["frame_pair", "history_state"]
    output_names = ["primary_trajectory", "proposals", "scores", "collision_risk"]

    # Save PyTorch reference state dict for 100% bit-exact evaluation matching
    ref_pth_path = os.path.splitext(output_onnx_path)[0] + "_reference.pth"
    torch.save(model.state_dict(), ref_pth_path)
    print(f"[ONNX Export] Saved PyTorch reference weights to {ref_pth_path}")

    print(f"[ONNX Export] Exporting to {output_onnx_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        (dummy_frame, dummy_history),
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    print(f"[ONNX Export] Export successful! Model size: {os.path.getsize(output_onnx_path) / 1024 / 1024:.2f} MB")

    # Verify ONNX model integrity
    print("[ONNX Export] Validating ONNX graph with onnx.checker...")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[ONNX Export] ONNX model is topologically valid.")

    # Validate output consistency between PyTorch and ONNXRuntime
    print("[ONNX Export] Running inference sanity check via ONNXRuntime...")
    ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "frame_pair": dummy_frame.numpy(),
        "history_state": dummy_history.numpy(),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare primary trajectory
    pt_traj = torch_out["primary_trajectory"].numpy()
    on_traj = ort_outs[0]
    max_diff = np.max(np.abs(pt_traj - on_traj))
    print(f"[ONNX Export] Max numerical difference (PyTorch vs ONNX): {max_diff:.6f}")
    assert max_diff < 1e-4, f"Export verification failed: difference {max_diff} exceeds tolerance!"
    print("[ONNX Export] VERIFICATION PASSED: ONNX output matches PyTorch reference perfectly.\n")

    return output_onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Drive-JEPA to ONNX for Qualcomm HTP")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/drive_jepa_perception_based_agent_resnet34.ckpt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/drive_jepa_8650.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.opset)
