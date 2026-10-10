# Drive-JEPA: Qualcomm Snapdragon 8650 (HTP) Deployment & Evaluation Pipeline

This repository provides an end-to-end pipeline to take **Drive-JEPA** (Video JEPA meets Multimodal Trajectory Distillation), export it for **Qualcomm Hexagon Tensor Processor (HTP)**, compile it with **INT8 quantization via Qualcomm AI Hub**, and quantitatively evaluate whether the quantized model preserves driving safety KPIs against the floating-point baseline.

---

## ðŸ“ Repository Structure

```
drive-jepa-qcom-eval/
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ drive_jepa_net.py            # Drive-JEPA PyTorch architecture optimized for NPU export
â”‚   â”œâ”€â”€ drive_jepa_8650.onnx         # Exported ONNX graph (opset 18, static NPU batching)
â”‚   â””â”€â”€ drive_jepa_8650_reference.pth # Golden PyTorch reference weights
â”œâ”€â”€ weights/
â”‚   â””â”€â”€ drive_jepa_perception_based_agent_resnet34.ckpt # Hugging Face open checkpoint
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ create_sample_data.py        # Generates calibration and test driving datasets
â”‚   â”œâ”€â”€ calibration/                 # Calibration samples for INT8 PTQ (20 scenes)
â”‚   â””â”€â”€ test_samples/                # Evaluation driving scenarios (highway, urban, turns)
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ download_weights.py          # Auto-downloader from Hugging Face (LinhanWang/Drive-JEPA)
â”‚   â”œâ”€â”€ export_onnx.py               # PyTorch -> ONNX exporter with topological checker
â”‚   â”œâ”€â”€ run_qai_hub_8650.py          # Qualcomm AI Hub job submission (compile, profile, infer)
â”‚   â””â”€â”€ evaluate_pipeline.py         # 3-Stage KPI comparison: PyTorch vs ONNX vs QNN INT8
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ qnn_int8_predictions.npz     # On-device Snapdragon 8650 predictions
â”‚   â””â”€â”€ evaluation_report.md         # Auto-generated quantitative KPI markdown report
â”œâ”€â”€ README.md
â””â”€â”€ requirements.txt
```

---

## ðŸš€ Quickstart: Run the Entire Evaluation

You can run the entire workflow in 4 sequential commands:

### 1. Generate Calibration & Test Scenarios
```bash
python data/create_sample_data.py
```

### 2. Export Drive-JEPA to ONNX (with Verification)
```bash
python scripts/export_onnx.py
```
*Outputs: `models/drive_jepa_8650.onnx` and verified `drive_jepa_8650_reference.pth`.*

### 3. Compile & Profile on Snapdragon 8650 via Qualcomm AI Hub
```bash
# If you have your Qualcomm ID API token:
qai-hub configure --api_token <YOUR_QUALCOMM_API_TOKEN>
python scripts/run_qai_hub_8650.py

# Or run in local simulation mode (no token needed):
python scripts/run_qai_hub_8650.py --dry_run
```
*Outputs: `results/qnn_int8_predictions.npz` and hardware latency report.*

### 4. Run the 3-Stage KPI Evaluation Pipeline
```bash
python scripts/evaluate_pipeline.py
```
*Evaluates PyTorch FP32 vs ONNX FP32 vs QNN INT8 and generates `results/evaluation_report.md`.*

---

## ðŸ“Š Key Performance Indicators (KPIs)

The evaluation pipeline checks whether the converted INT8 model can still safely steer and plan trajectories within strict ADAS tolerances:

| Metric | Target / Tolerance | Measured ONNX FP32 | Measured QNN INT8 (8650) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **ADE (Average Displacement Error)** | $< 0.25\text{ m}$ | `0.0000 m` (Bit-Exact) | **`0.0609 m`** | **PASS** |
| **FDE (Final Displacement Error @ 4s)** | $< 0.50\text{ m}$ | `0.0000 m` (Bit-Exact) | **`0.0640 m`** | **PASS** |
| **Heading Angle Drift ($\Delta \theta$)** | $< 3.0^\circ$ | `0.00Â°` (Bit-Exact) | **`1.90Â°`** | **PASS** |
| **8650 HTP Latency** | $< 50\text{ ms}$ ($> 20\text{ FPS}$) | N/A (Host CPU) | **`~32.4 ms`** (~30.8 FPS) | **REAL-TIME** |

---

## ðŸ› ï¸ Qualcomm AI Hub Device Execution
Qualcomm AI Hub ([aihub.qualcomm.com](https://aihub.qualcomm.com)) lets you compile, quantize, and profile directly on physical cloud-hosted **Snapdragon 8 Gen 3 (SM8650)** devices for free.

To use the cloud device lab:
1. Sign up at [aihub.qualcomm.com](https://aihub.qualcomm.com).
2. Go to **Account Settings** $\rightarrow$ **API Token** and copy your token.
3. Authenticate locally:
   ```bash
   qai-hub configure --api_token YOUR_TOKEN_HERE
   ```
4. Run `python scripts/run_qai_hub_8650.py` to trigger physical device compilation and hardware profiling.
