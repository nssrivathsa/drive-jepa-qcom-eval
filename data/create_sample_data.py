import os
import numpy as np


def generate_synthetic_driving_data(output_dir: str = "data", num_samples: int = 20):
    """
    Generates synthetic driving data (frames + history states) for:
    - Quantization calibration dataset (PTQ calibration)
    - Golden evaluation test dataset (accuracy & fidelity verification)
    """
    calib_dir = os.path.join(output_dir, "calibration")
    test_dir = os.path.join(output_dir, "test_samples")
    os.makedirs(calib_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.random.seed(42)

    print(f"[SampleData] Generating {num_samples} calibration samples in {calib_dir}...")
    for i in range(num_samples):
        # Frame pair: [1, 6, 256, 512] representing consecutive driving video frames (normalized)
        # Add road-like structure (gradient for road surface + simulated objects)
        base_frame = np.random.uniform(-1.0, 1.0, (1, 6, 256, 512)).astype(np.float32)
        # Add vertical gradient simulating road perspective
        gradient = np.linspace(-0.5, 0.5, 256).reshape(1, 1, 256, 1)
        base_frame[:, :, :, :] += gradient

        # History state: [1, 11]
        # [vx, vy, yaw_rate, ax, ay, past_x1, past_y1, past_x2, past_y2, past_x3, past_y3]
        speed_mps = np.random.uniform(5.0, 30.0)  # 18 - 108 km/h
        yaw_rate = np.random.uniform(-0.1, 0.1)
        history = np.array([
            speed_mps, 0.0, yaw_rate, 0.5, 0.0,
            -2.0 * speed_mps * 0.1, 0.0,
            -4.0 * speed_mps * 0.1, 0.0,
            -6.0 * speed_mps * 0.1, 0.0,
        ], dtype=np.float32).reshape(1, 11)

        calib_file = os.path.join(calib_dir, f"sample_{i:03d}.npz")
        np.savez(calib_file, frame_pair=base_frame, history_state=history)

    print(f"[SampleData] Generating test evaluation samples in {test_dir}...")
    test_scenarios = [
        ("straight_highway", 25.0, 0.0),    # 90 km/h cruising
        ("gentle_left_turn", 18.0, 0.05),   # 65 km/h left curve
        ("gentle_right_turn", 18.0, -0.05), # 65 km/h right curve
        ("urban_low_speed", 8.0, 0.0),      # 28 km/h city driving
        ("braking_scenario", 12.0, 0.0),    # decelerating
    ]

    for name, spd, yr in test_scenarios:
        frame = np.random.uniform(-1.0, 1.0, (1, 6, 256, 512)).astype(np.float32)
        history = np.array([
            spd, 0.0, yr, -1.0 if "braking" in name else 0.2, 0.0,
            -spd * 0.2, 0.0,
            -spd * 0.4, 0.0,
            -spd * 0.6, 0.0,
        ], dtype=np.float32).reshape(1, 11)

        test_file = os.path.join(test_dir, f"{name}.npz")
        np.savez(test_file, frame_pair=frame, history_state=history)

    print("[SampleData] Successfully generated driving calibration and test datasets.")


if __name__ == "__main__":
    generate_synthetic_driving_data()
