import os
import sys
import glob
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.drive_jepa_net import load_drive_jepa_model


def generate_visualizations(
    checkpoint_path: str = "models/drive_jepa_8650_reference.pth",
    onnx_path: str = "models/drive_jepa_8650.onnx",
    qnn_results_path: str = "results/qnn_int8_predictions.npz",
    test_dir: str = "data/test_samples",
    output_dir: str = "results/visualizations",
):
    """
    Renders high-resolution BEV trajectory visualizations and error distribution charts:
    - Compares PyTorch FP32 vs ONNX FP32 vs Qualcomm QNN INT8 (8650 HTP)
    - Saves individual scenario plots and consolidated dashboards
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("qai_hub_results/visualizations", exist_ok=True)

    print("[Visualizer] Loading models for trajectory visualization...")
    pt_model = load_drive_jepa_model(checkpoint_path=checkpoint_path, device="cpu")
    pt_model.eval()

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    qnn_data = np.load(qnn_results_path) if os.path.exists(qnn_results_path) else None

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    if not test_files:
        print("[Visualizer] No test samples found.")
        return

    fig, axes = plt.subplots(1, len(test_files), figsize=(20, 5), dpi=200)
    if len(test_files) == 1:
        axes = [axes]

    error_summary = {}

    for idx, tf in enumerate(test_files):
        scenario = os.path.splitext(os.path.basename(tf))[0]
        data = np.load(tf)
        frame_pair = data["frame_pair"]
        history = data["history_state"]

        # 1. PyTorch Forward
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(frame_pair), torch.from_numpy(history))
        pt_traj = pt_out["primary_trajectory"][0].numpy()  # [8, 3]

        # 2. ONNX Forward
        ort_out = ort_session.run(None, {"frame_pair": frame_pair, "history_state": history})
        onnx_traj = ort_out[0][0]

        # 3. QNN INT8
        qnn_traj = None
        if qnn_data is not None and scenario in qnn_data:
            qnn_traj = qnn_data[scenario]
            if qnn_traj.ndim == 3:
                qnn_traj = qnn_traj[0]

        # Trajectory coordinates: (x forward in meters, y lateral in meters)
        ax = axes[idx]
        ax.set_facecolor("#1a1a24")

        # Plot road boundaries / safety corridor
        y_center = pt_traj[:, 1]
        x_forward = pt_traj[:, 0]
        ax.fill_betweenx(x_forward, y_center - 1.75, y_center + 1.75, color="#2e3440", alpha=0.6, label="Lane Corridor (3.5m)")
        ax.fill_betweenx(x_forward, y_center - 0.25, y_center + 0.25, color="#4c566a", alpha=0.5, label="Tolerance (±0.25m)")

        # Plot paths
        ax.plot(pt_traj[:, 1], pt_traj[:, 0], color="#a3be8c", linewidth=2.5, label="PyTorch FP32 (Golden)")
        ax.plot(onnx_traj[:, 1], onnx_traj[:, 0], color="#88c0d0", linestyle="--", linewidth=2.0, label="ONNX FP32")

        if qnn_traj is not None:
            ax.plot(qnn_traj[:, 1], qnn_traj[:, 0], color="#ebcb8b", linestyle=":", marker="o", markersize=4, linewidth=2.2, label="QNN INT8 (8650)")
            # Compute step errors
            diff = np.linalg.norm(qnn_traj[:, :2] - pt_traj[:, :2], axis=-1)
            error_summary[scenario] = diff

        ax.set_title(f"Scenario: {scenario}", fontsize=11, fontweight="bold", color="#eceff4")
        ax.set_xlabel("Lateral Y (m)", fontsize=9, color="#d8dee9")
        ax.set_ylabel("Forward X (m)", fontsize=9, color="#d8dee9")
        ax.tick_params(colors="#d8dee9")
        ax.grid(True, linestyle="--", alpha=0.3, color="#4c566a")
        ax.set_xlim([-4.0, 4.0])
        ax.set_ylim([0, 45])
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7, facecolor="#2e3440", edgecolor="#4c566a", labelcolor="#eceff4")

    plt.suptitle("Drive-JEPA Trajectory Fidelity: PyTorch FP32 vs ONNX FP32 vs Qualcomm QNN INT8 (8650 HTP)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    dashboard_path = os.path.join(output_dir, "trajectory_comparison_bev.png")
    fig.savefig(dashboard_path, bbox_inches="tight")
    fig.savefig("qai_hub_results/visualizations/trajectory_comparison_bev.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Saved BEV dashboard to: {dashboard_path}")

    # Generate Error Distribution Chart
    if error_summary:
        fig_err, ax_err = plt.subplots(figsize=(10, 4), dpi=200)
        ax_err.set_facecolor("#1a1a24")
        time_horizons = [f"{t:.1f}s" for t in np.linspace(0.5, 4.0, 8)]

        for sc_name, errors in error_summary.items():
            ax_err.plot(time_horizons, errors, marker="s", markersize=5, linewidth=1.8, label=sc_name)

        ax_err.axhline(0.25, color="#bf616a", linestyle="--", linewidth=1.5, label="ADAS Max Tolerance (0.25m)")
        ax_err.set_title("INT8 Quantization Error Across 4.0s Planning Horizon", fontsize=12, fontweight="bold", color="#eceff4")
        ax_err.set_xlabel("Planning Horizon (Seconds)", fontsize=10, color="#d8dee9")
        ax_err.set_ylabel("Displacement Error (Meters)", fontsize=10, color="#d8dee9")
        ax_err.tick_params(colors="#d8dee9")
        ax_err.grid(True, linestyle="--", alpha=0.3, color="#4c566a")
        ax_err.legend(loc="upper left", fontsize=8, facecolor="#2e3440", edgecolor="#4c566a", labelcolor="#eceff4")
        ax_err.set_ylim([0, 0.35])

        err_chart_path = os.path.join(output_dir, "int8_error_distribution.png")
        fig_err.savefig(err_chart_path, bbox_inches="tight")
        fig_err.savefig("qai_hub_results/visualizations/int8_error_distribution.png", bbox_inches="tight")
        plt.close(fig_err)
        print(f"[Visualizer] Saved error chart to: {err_chart_path}")

    print("[Visualizer] Visualizations successfully generated!\n")


if __name__ == "__main__":
    generate_visualizations()
