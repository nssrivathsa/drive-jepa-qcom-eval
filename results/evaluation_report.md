# Drive-JEPA Qualcomm Snapdragon 8650 Evaluation Report

## 1. Executive Summary
- **Model**: Drive-JEPA Spatio-Temporal Planning Model
- **Target Chip**: Qualcomm Snapdragon 8 Gen 3 / SA8650 (Hexagon Tensor Processor / HTP)
- **Quantization**: INT8 Post-Training Quantization (PTQ)
- **Throughput**: **~30.8 FPS** (~32.4 ms latency) on 8650 HTP
- **Fidelity Verdict**: **PASSED** (INT8 trajectory closely matches FP32 golden baseline within safety tolerances)

## 2. Quantitative KPI Comparison Table

| Test Scenario | Runtime Stage | ADE (m) | FDE (m) | Heading Error (°) | Max Lateral Error (m) | Safety Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **braking_scenario** | ONNX FP32 | 0.0000 | 0.0000 | 0.0° | 0.0 | Bit-Exact |
| | **QNN INT8 (8650)** | **0.0708** | **0.0877** | **1.98°** | **0.0726** | **PASS** |
| **gentle_left_turn** | ONNX FP32 | 0.0000 | 0.0000 | 0.0° | 0.0 | Bit-Exact |
| | **QNN INT8 (8650)** | **0.0619** | **0.0937** | **2.68°** | **0.0806** | **PASS** |
| **gentle_right_turn** | ONNX FP32 | 0.0000 | 0.0000 | 0.0° | 0.0 | Bit-Exact |
| | **QNN INT8 (8650)** | **0.0524** | **0.0721** | **1.2°** | **0.0586** | **PASS** |
| **straight_highway** | ONNX FP32 | 0.0000 | 0.0000 | 0.0° | 0.0 | Bit-Exact |
| | **QNN INT8 (8650)** | **0.0554** | **0.0112** | **1.52°** | **0.0877** | **PASS** |
| **urban_low_speed** | ONNX FP32 | 0.0000 | 0.0000 | 0.0° | 0.0 | Bit-Exact |
| | **QNN INT8 (8650)** | **0.0639** | **0.0552** | **2.14°** | **0.0809** | **PASS** |

## 3. ADAS Safety Thresholds and Analysis
- **Average Displacement Error (ADE)**: Measured `~0.061 m` against a safety threshold of `< 0.25 m`.
- **Final Displacement Error (FDE at 4s)**: Measured `~0.064 m` against a safety threshold of `< 0.50 m`.
- **Heading Drift**: Measured `~1.90°`, well inside the standard steering servo deadband.
- **Lane Keeping**: Max lateral deviation remained under 0.15 m, confirming that INT8 quantization does not cause lane departure or trajectory oscillation.
