"""Task 1: Bundle Adjustment from scratch with PyTorch.

This script estimates:
1. A shared focal length.
2. Camera extrinsics for all views.
3. A 3D point cloud for all observed points.

It uses the 2D correspondences from ``data/points2d.npz`` and exports a
colored OBJ file that can be inspected in MeshLab or Blender.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "PyTorch is required for Task 1. Install it first, then rerun this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Bundle Adjustment with PyTorch")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root_dir / "data",
        help="Directory that contains points2d.npz and points3d_colors.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "outputs" / "task1",
        help="Directory used to save optimized outputs",
    )
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=1024)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=600,
        help="Number of Adam optimization steps",
    )
    parser.add_argument(
        "--camera-lr",
        type=float,
        default=1e-2,
        help="Learning rate for camera rotations and translations",
    )
    parser.add_argument(
        "--point-lr",
        type=float,
        default=5e-3,
        help="Learning rate for 3D point coordinates",
    )
    parser.add_argument(
        "--focal-lr",
        type=float,
        default=5e-3,
        help="Learning rate for the focal length",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="Initial camera orbit radius",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=60.0,
        help="Initial field of view used to derive the focal length",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print progress every N steps",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for optimization",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating-point precision",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-loss-plot",
        action="store_true",
        help="Save a loss curve if matplotlib is available",
    )
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but no CUDA device is available.")

    return torch.device(device_name)


def normalize_vectors(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vectors / (vectors.norm(dim=-1, keepdim=True) + eps)


def look_at_world_to_camera(centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros_like(centers)
    up_guess = torch.tensor(
        [0.0, 1.0, 0.0], dtype=centers.dtype, device=centers.device
    ).expand_as(centers)

    forward = normalize_vectors(target - centers)
    right = normalize_vectors(torch.cross(forward, up_guess, dim=-1))
    up = normalize_vectors(torch.cross(right, forward, dim=-1))
    backward = -forward

    rotations = torch.stack([right, up, backward], dim=1)
    translations = -(rotations @ centers.unsqueeze(-1)).squeeze(-1)
    return rotations, translations


def matrix_to_euler_xyz(rotations: torch.Tensor) -> torch.Tensor:
    sy = torch.sqrt(rotations[:, 0, 0] ** 2 + rotations[:, 1, 0] ** 2)
    singular = sy < 1e-6

    x = torch.atan2(rotations[:, 2, 1], rotations[:, 2, 2])
    y = torch.atan2(-rotations[:, 2, 0], sy)
    z = torch.atan2(rotations[:, 1, 0], rotations[:, 0, 0])

    x_singular = torch.atan2(-rotations[:, 1, 2], rotations[:, 1, 1])
    y_singular = torch.atan2(-rotations[:, 2, 0], sy)
    z_singular = torch.zeros_like(z)

    x = torch.where(singular, x_singular, x)
    y = torch.where(singular, y_singular, y)
    z = torch.where(singular, z_singular, z)
    return torch.stack([x, y, z], dim=-1)


def euler_xyz_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    x, y, z = euler_angles.unbind(-1)
    cx = torch.cos(x)
    cy = torch.cos(y)
    cz = torch.cos(z)
    sx = torch.sin(x)
    sy = torch.sin(y)
    sz = torch.sin(z)

    row0 = torch.stack(
        [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy], dim=-1
    )
    row1 = torch.stack(
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx], dim=-1
    )
    row2 = torch.stack([-sy, cy * sx, cx * cy], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def initial_camera_setup(
    num_views: int,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.linspace(0.0, 2.0 * math.pi, num_views + 1, device=device, dtype=dtype)
    angles = angles[:-1]
    centers = torch.stack(
        [
            radius * torch.sin(angles),
            torch.zeros_like(angles),
            radius * torch.cos(angles),
        ],
        dim=-1,
    )
    return look_at_world_to_camera(centers)


def triangulate_points(
    observations_xy: torch.Tensor,
    visibility_mask: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    focal_length: torch.Tensor,
    cx: float,
    cy: float,
) -> torch.Tensor:
    num_points = observations_xy.shape[1]
    device = rotations.device
    dtype = rotations.dtype

    normal_matrices = torch.zeros((num_points, 4, 4), dtype=dtype, device=device)
    projection_matrices = torch.cat([rotations, translations.unsqueeze(-1)], dim=-1)

    for view_idx in range(rotations.shape[0]):
        visible = visibility_mask[view_idx]
        if not torch.any(visible):
            continue

        uv = observations_xy[view_idx]
        p1 = projection_matrices[view_idx, 0].unsqueeze(0)
        p2 = projection_matrices[view_idx, 1].unsqueeze(0)
        p3 = projection_matrices[view_idx, 2].unsqueeze(0)

        row1 = focal_length * p1 + (uv[:, 0:1] - cx) * p3
        row2 = -focal_length * p2 + (uv[:, 1:2] - cy) * p3

        rows = torch.stack([row1, row2], dim=1)[visible]
        normal_matrices[visible] += torch.matmul(rows.transpose(-1, -2), rows)

    normal_matrices += 1e-6 * torch.eye(4, dtype=dtype, device=device).unsqueeze(0)
    _, eigenvectors = torch.linalg.eigh(normal_matrices)
    homogeneous_points = eigenvectors[:, :, 0]
    w = homogeneous_points[:, 3:4]
    safe_w = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8), w)
    return homogeneous_points[:, :3] / safe_w


def project_points(
    points3d: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    focal_length: torch.Tensor,
    cx: float,
    cy: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_points = torch.einsum("vij,nj->vni", rotations, points3d)
    camera_points = camera_points + translations[:, None, :]

    depth = camera_points[..., 2]
    depth_sign = torch.where(depth >= 0.0, torch.ones_like(depth), -torch.ones_like(depth))
    safe_depth = depth_sign * torch.clamp(depth.abs(), min=1e-4)

    projected_u = -focal_length * camera_points[..., 0] / safe_depth + cx
    projected_v = focal_length * camera_points[..., 1] / safe_depth + cy
    projections = torch.stack([projected_u, projected_v], dim=-1)
    return projections, camera_points


def save_colored_obj(path: Path, points3d: np.ndarray, colors: np.ndarray) -> None:
    colors = np.clip(colors, 0.0, 1.0)
    with path.open("w", encoding="utf-8") as file:
        for point, color in zip(points3d, colors):
            file.write(
                "v "
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def save_loss_plot(path: Path, loss_history: list[float]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(loss_history, linewidth=2)
    axis.set_title("Bundle Adjustment Loss")
    axis.set_xlabel("Optimization Step")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return True


def load_data(
    data_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor, torch.Tensor, np.ndarray]:
    points2d_path = data_dir / "points2d.npz"
    colors_path = data_dir / "points3d_colors.npy"

    if not points2d_path.exists():
        raise FileNotFoundError(f"Missing file: {points2d_path}")
    if not colors_path.exists():
        raise FileNotFoundError(f"Missing file: {colors_path}")

    points2d = np.load(points2d_path)
    view_names = sorted(points2d.files)
    observations = np.stack([points2d[name] for name in view_names], axis=0).astype(np.float32)
    colors = np.load(colors_path).astype(np.float32)

    observations_xy = torch.tensor(observations[..., :2], dtype=dtype, device=device)
    visibility_mask = torch.tensor(observations[..., 2] > 0.5, device=device)
    return view_names, observations_xy, visibility_mask, colors


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dtype = getattr(torch, args.dtype)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    view_names, observations_xy, visibility_mask, colors = load_data(args.data_dir, device, dtype)
    num_views, num_points = observations_xy.shape[:2]
    cx = args.image_width / 2.0
    cy = args.image_height / 2.0

    initial_rotations, initial_translations = initial_camera_setup(
        num_views=num_views,
        radius=args.radius,
        device=device,
        dtype=dtype,
    )
    initial_focal = torch.tensor(
        args.image_height / (2.0 * math.tan(math.radians(args.fov_deg) / 2.0)),
        dtype=dtype,
        device=device,
    )
    initial_points = triangulate_points(
        observations_xy=observations_xy,
        visibility_mask=visibility_mask,
        rotations=initial_rotations,
        translations=initial_translations,
        focal_length=initial_focal,
        cx=cx,
        cy=cy,
    )

    fixed_rotation = initial_rotations[:1].detach().clone()
    fixed_translation = initial_translations[:1].detach().clone()

    camera_euler = torch.nn.Parameter(
        matrix_to_euler_xyz(initial_rotations[1:]).detach().clone()
    )
    camera_translations = torch.nn.Parameter(initial_translations[1:].detach().clone())
    points3d = torch.nn.Parameter(initial_points.detach().clone())
    log_focal = torch.nn.Parameter(torch.log(initial_focal).detach().clone())

    optimizer = torch.optim.Adam(
        [
            {"params": [camera_euler, camera_translations], "lr": args.camera_lr},
            {"params": [points3d], "lr": args.point_lr},
            {"params": [log_focal], "lr": args.focal_lr},
        ]
    )

    loss_history: list[float] = []
    reprojection_history: list[float] = []

    print(
        f"Running bundle adjustment on {device.type} "
        f"with {num_views} views and {num_points} points."
    )

    for step in range(1, args.num_steps + 1):
        optimizer.zero_grad()

        rotations = torch.cat([fixed_rotation, euler_xyz_to_matrix(camera_euler)], dim=0)
        translations = torch.cat([fixed_translation, camera_translations], dim=0)
        focal_length = torch.exp(log_focal)

        projected_xy, camera_points = project_points(
            points3d=points3d,
            rotations=rotations,
            translations=translations,
            focal_length=focal_length,
            cx=cx,
            cy=cy,
        )

        residual = projected_xy - observations_xy
        visible_residual = residual[visibility_mask]
        visible_depth = camera_points[..., 2][visibility_mask]

        reprojection_loss = F.huber_loss(
            visible_residual,
            torch.zeros_like(visible_residual),
            reduction="mean",
            delta=10.0,
        )
        depth_penalty = torch.relu(visible_depth + 1e-3).mean()
        center_penalty = points3d.mean(dim=0).pow(2).mean()
        radius_penalty = ((translations[1:].norm(dim=-1) - args.radius) ** 2).mean()
        focal_penalty = ((focal_length - initial_focal) / initial_focal).pow(2)

        loss = (
            reprojection_loss
            + 0.1 * depth_penalty
            + 1e-3 * center_penalty
            + 1e-3 * radius_penalty
            + 1e-3 * focal_penalty
        )
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.detach().cpu()))
        reprojection_history.append(float(reprojection_loss.detach().cpu()))

        if step == 1 or step % args.print_every == 0 or step == args.num_steps:
            pixel_rmse = torch.sqrt((visible_residual**2).sum(dim=-1).mean()).item()
            print(
                f"[{step:04d}/{args.num_steps:04d}] "
                f"loss={loss.item():.4f} "
                f"reproj={reprojection_loss.item():.4f} "
                f"rmse={pixel_rmse:.4f}px "
                f"f={focal_length.item():.2f}"
            )

    final_rotations = torch.cat(
        [fixed_rotation, euler_xyz_to_matrix(camera_euler)], dim=0
    ).detach()
    final_translations = torch.cat([fixed_translation, camera_translations], dim=0).detach()
    final_focal = torch.exp(log_focal).detach()
    final_points = points3d.detach()

    final_projected_xy, _ = project_points(
        points3d=final_points,
        rotations=final_rotations,
        translations=final_translations,
        focal_length=final_focal,
        cx=cx,
        cy=cy,
    )
    final_visible_residual = (final_projected_xy - observations_xy)[visibility_mask]
    final_pixel_rmse = torch.sqrt((final_visible_residual**2).sum(dim=-1).mean()).item()

    result_npz_path = output_dir / "bundle_adjustment_result.npz"
    result_obj_path = output_dir / "reconstructed_points.obj"
    summary_json_path = output_dir / "summary.json"

    np.savez(
        result_npz_path,
        points3d=final_points.cpu().numpy(),
        colors=colors,
        view_names=np.array(view_names),
        rotations=final_rotations.cpu().numpy(),
        translations=final_translations.cpu().numpy(),
        focal_length=np.array([final_focal.item()], dtype=np.float32),
        loss_history=np.array(loss_history, dtype=np.float32),
        reprojection_history=np.array(reprojection_history, dtype=np.float32),
    )
    save_colored_obj(result_obj_path, final_points.cpu().numpy(), colors)

    summary = {
        "device": device.type,
        "num_views": num_views,
        "num_points": num_points,
        "num_steps": args.num_steps,
        "focal_length_px": float(final_focal.item()),
        "final_loss": float(loss_history[-1]),
        "final_reprojection_loss": float(reprojection_history[-1]),
        "final_pixel_rmse": float(final_pixel_rmse),
        "output_npz": str(result_npz_path),
        "output_obj": str(result_obj_path),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_loss_plot:
        saved = save_loss_plot(output_dir / "loss_curve.png", loss_history)
        if not saved:
            print("matplotlib is not installed, so the loss curve was not saved.")

    print("Bundle adjustment finished.")
    print(f"Saved results to: {result_npz_path}")
    print(f"Saved colored OBJ to: {result_obj_path}")
    print(f"Saved summary to: {summary_json_path}")
    print(
        "Final metrics: "
        f"reprojection={reprojection_history[-1]:.4f}, "
        f"rmse={final_pixel_rmse:.4f}px, "
        f"focal={final_focal.item():.2f}px"
    )


if __name__ == "__main__":
    main()
