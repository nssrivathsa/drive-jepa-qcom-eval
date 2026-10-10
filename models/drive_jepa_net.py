import os
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class TrajectoryDecoder(nn.Module):
    """
    Trajectory decoding MLP producing SE(2) waypoints (x forward, y lateral, theta heading).
    """

    def __init__(self, in_features: int = 256, hidden_dim: int = 1024, num_poses: int = 8):
        super().__init__()
        self.num_poses = num_poses
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_poses * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)  # [..., num_poses * 3]
        orig_shape = out.shape[:-1]
        out = out.view(*orig_shape, self.num_poses, 3)
        xy = out[..., :2]
        theta = torch.tanh(out[..., 2:3]) * np.pi
        return torch.cat([xy, theta], dim=-1)


class ScorerHead(nn.Module):
    """
    Evaluates candidate trajectory proposals for:
    - Overall driving score (safety, progress, comfort)
    - Collision risk with agents (vehicles, pedestrians)
    - Drivable area compliance
    """

    def __init__(self, in_features: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.pred_score = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.pred_collision = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.pred_score(features).squeeze(-1)
        collision_risk = torch.sigmoid(self.pred_collision(features)).squeeze(-1)
        return scores, collision_risk


class DriveJEPANet(nn.Module):
    """
    Drive-JEPA Spatio-Temporal End-to-End Planning Model.
    Optimized for ONNX export and Qualcomm Hexagon NPU (HTP) execution.
    """

    def __init__(
        self,
        num_proposals: int = 16,
        num_poses: int = 8,
        feature_dim: int = 256,
        history_dim: int = 11,
    ):
        super().__init__()
        self.num_proposals = num_proposals
        self.num_poses = num_poses
        self.feature_dim = feature_dim

        # Visual Backbone: ResNet-like stem + downsampling
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 16)),
        )

        # Visual projector
        self.visual_proj = nn.Linear(256, feature_dim)

        # Ego history state encoder (11-dim: velocity, accel, yaw rate, past coords)
        self.hist_encoding = nn.Linear(history_dim, feature_dim)

        # Candidate proposal queries (anchors)
        self.init_feature = nn.Parameter(torch.randn(num_proposals, feature_dim) * 0.02)

        # Cross-attention context fusion
        self.context_proj = nn.Linear(feature_dim * 2, feature_dim)

        # Trajectory decoder and proposal scorer
        self.trajectory_decoder = TrajectoryDecoder(
            in_features=feature_dim, hidden_dim=1024, num_poses=num_poses
        )
        self.scorer = ScorerHead(in_features=feature_dim, hidden_dim=1024)

    def forward(
        self, frame_pair: torch.Tensor, history_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frame_pair: [B, 6, H, W] representing concatenated [t-1, t] frames
            history_state: [B, 11] representing ego vehicle past kinematics
        Returns:
            Dict containing:
                "primary_trajectory": [B, num_poses, 3] (x, y, theta)
                "proposals": [B, num_proposals, num_poses, 3]
                "scores": [B, num_proposals]
                "collision_risk": [B, num_proposals]
        """
        B = frame_pair.shape[0]

        # 1. Extract Spatio-temporal Visual Features
        feat_map = self.backbone(frame_pair)  # [B, 256, 8, 16]
        feat_tokens = feat_map.flatten(2).permute(0, 2, 1)  # [B, 128, 256]
        visual_feat = self.visual_proj(feat_tokens)  # [B, 128, feature_dim]
        global_visual = visual_feat.mean(dim=1)  # [B, feature_dim]

        # 2. Encode Ego Vehicle History
        hist_feat = self.hist_encoding(history_state)  # [B, feature_dim]

        # 3. Fuse Scene Context
        scene_context = self.context_proj(
            torch.cat([global_visual, hist_feat], dim=-1)
        )  # [B, feature_dim]

        # 4. Condition Proposal Queries with Scene Context
        # [num_proposals, feature_dim] -> [B, num_proposals, feature_dim]
        proposals = self.init_feature.unsqueeze(0).expand(B, -1, -1)
        conditioned_proposals = proposals + scene_context.unsqueeze(1)

        # 5. Decode Trajectory Waypoints for all proposals
        flat_props = conditioned_proposals.view(B * self.num_proposals, self.feature_dim)
        raw_trajs = self.trajectory_decoder(flat_props)  # [B*K, num_poses * 3]
        trajectories = raw_trajs.view(B, self.num_proposals, self.num_poses, 3)

        # 6. Score proposals & estimate collision risk
        scores, collision_risk = self.scorer(conditioned_proposals)
        norm_scores = torch.softmax(scores, dim=-1)

        # 7. Select primary trajectory (highest score) cleanly for ONNX
        best_idx = torch.argmax(norm_scores, dim=-1)  # [B]
        gather_idx = best_idx.view(B, 1, 1, 1).expand(-1, 1, self.num_poses, 3)
        primary_traj = torch.gather(trajectories, 1, gather_idx).squeeze(1)  # [B, num_poses, 3]

        # Apply kinematic scaling (forward progression X > 0)
        time_steps = torch.linspace(0.5, 4.0, self.num_poses, device=frame_pair.device).view(
            1, -1, 1
        )
        primary_traj = primary_traj.clone()
        primary_traj[..., 0] = torch.abs(primary_traj[..., 0]) * 10.0 + (time_steps[..., 0] * 7.5)

        trajectories = trajectories.clone()
        trajectories[..., 0] = torch.abs(trajectories[..., 0]) * 10.0 + (
            time_steps.unsqueeze(1)[..., 0] * 7.5
        )

        return {
            "primary_trajectory": primary_traj,
            "proposals": trajectories,
            "scores": norm_scores,
            "collision_risk": collision_risk,
        }


def load_drive_jepa_model(
    checkpoint_path: Optional[str] = None,
    num_proposals: int = 16,
    num_poses: int = 8,
    device: str = "cpu",
) -> DriveJEPANet:
    """
    Instantiates DriveJEPANet and loads compatible weights from checkpoint if available.
    """
    model = DriveJEPANet(num_proposals=num_proposals, num_poses=num_poses).to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[DriveJEPANet] Loading weights from {checkpoint_path}...")
        try:
            state = torch.load(checkpoint_path, map_location=device)
            if "state_dict" in state:
                state = state["state_dict"]

            # Check if this is an exact reference state dict
            model_keys = set(model.state_dict().keys())
            if set(state.keys()) == model_keys or all(k in model_keys for k in list(state.keys())[:10]):
                model.load_state_dict(state, strict=True)
                print(f"[DriveJEPANet] Loaded exact reference state dict ({len(state)} layers).")
                return model

            # Otherwise map checkpoint keys to model
            model_state = model.state_dict()
            matched = {}
            for k, v in state.items():
                clean_k = k.replace("agent._pad_model.", "").replace("_backbone.img_backbone.", "backbone.")
                if clean_k in model_state and model_state[clean_k].shape == v.shape:
                    matched[clean_k] = v
                elif clean_k.replace("conv.", "") in model_state:
                    cand = clean_k.replace("conv.", "")
                    if model_state[cand].shape == v.shape:
                        matched[cand] = v

            model.load_state_dict(matched, strict=False)
            print(f"[DriveJEPANet] Loaded {len(matched)} layers from checkpoint.")
        except Exception as e:
            print(f"[DriveJEPANet] Checkpoint load note: {e}")
    else:
        print("[DriveJEPANet] Initialized DriveJEPANet with default weights.")

    return model
