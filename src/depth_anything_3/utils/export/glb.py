# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .depth_vis import export_to_depth_vis


def _extract_pitch_and_roll_from_exif(image_path: str) -> tuple[Optional[float], Optional[float]]:
    """
    Extract both pitch and roll angles from image EXIF metadata.
    Expected format in User Comment: "Pitch:26.736341,Roll:-92.33134,Azimuth:-24.44831"
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (pitch, roll) in degrees, or (None, None) if not found
    """
    try:
        import re
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if exif_data is None:
            return None, None
            
        # Look for pitch and roll in User Comment or other text fields
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            
            # Check UserComment, ImageDescription, and other text fields
            if tag in ['UserComment', 'ImageDescription', 'Comment', 'XPComment']:
                # Convert bytes to string if needed
                if isinstance(value, bytes):
                    try:
                        value_str = value.decode('utf-8', errors='ignore')
                    except:
                        value_str = str(value)
                else:
                    value_str = str(value)
                
                # Parse format: "Pitch:26.736341,Roll:-92.33134,Azimuth:-24.44831"
                pitch_match = re.search(r'Pitch:\s*(-?\d+\.?\d*)', value_str, re.IGNORECASE)
                roll_match = re.search(r'Roll:\s*(-?\d+\.?\d*)', value_str, re.IGNORECASE)
                
                pitch = float(pitch_match.group(1)) if pitch_match else None
                roll = float(roll_match.group(1)) if roll_match else None
                
                if pitch is not None or roll is not None:
                    return pitch, roll
                
        return None, None
    except Exception as e:
        logger.warning(f"Could not extract EXIF from {image_path}: {e}")
        return None, None


def _compute_simple_pitch_alignment(
    pitch_angle_degrees: float,
    roll_angle_degrees: float,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """
    Compute simple gravity alignment by first aligning roll, then rotating by pitch.
    
    Two-step process:
    1. Rotate around Z-axis (forward) to align roll to -90° (make X-axis horizontal)
    2. Rotate around the now-horizontal X-axis by the pitch angle
    
    Args:
        pitch_angle_degrees: Pitch angle in degrees from sensor (positive = looking down, negative = looking up)
        roll_angle_degrees: Roll angle in degrees from sensor
        extrinsics: Camera extrinsic matrices (N, 3, 4) or (N, 4, 4) - only first camera is used
        
    Returns:
        4x4 transformation matrix representing the combined roll and pitch alignment
    """
    num_cameras = len(extrinsics)
    
    # Ensure we have 4x4 matrices
    if extrinsics.shape[-2:] == (3, 4):
        ext_4x4 = np.zeros((num_cameras, 4, 4))
        ext_4x4[:, :3, :4] = extrinsics
        ext_4x4[:, 3, 3] = 1
    else:
        ext_4x4 = extrinsics
    
    # Get first camera's world-to-camera transform
    w2c = ext_4x4[0]
    try:
        c2w = np.linalg.inv(w2c)
    except np.linalg.LinAlgError:
        logger.warning("Could not invert first camera matrix for pitch alignment")
        return np.eye(4)
    
    # Extract first camera's axes in world coordinates
    R_c2w = c2w[:3, :3]
    x_axis_world = R_c2w @ np.array([1, 0, 0])  # Camera's X-axis (right) in world
    z_axis_world = R_c2w @ np.array([0, 0, 1])  # Camera's Z-axis (forward) in world
    
    # Normalize the axes
    x_axis_world = x_axis_world / np.linalg.norm(x_axis_world)
    z_axis_world = z_axis_world / np.linalg.norm(z_axis_world)
    
    logger.info(f"Two-step gravity alignment:")
    logger.info(f"  Camera X-axis (right) in world:   [{x_axis_world[0]:.3f}, {x_axis_world[1]:.3f}, {x_axis_world[2]:.3f}]")
    logger.info(f"  Camera Z-axis (forward) in world: [{z_axis_world[0]:.3f}, {z_axis_world[1]:.3f}, {z_axis_world[2]:.3f}]")
    
    # Step 1: Rotate around Z-axis (forward) to align roll to -90°
    # Target: make X-axis horizontal (zero Z-component)
    # We need to rotate to make roll = -90° (X-axis points right and horizontal)
    roll_compensation = -90.0 - roll_angle_degrees  # Amount to rotate to get to -90°
    roll_rotation_angle = -np.radians(roll_compensation)  # Negative for correct Z-axis rotation direction
    
    logger.info(f"  Step 1 - Roll alignment:")
    logger.info(f"    Current roll: {roll_angle_degrees:.2f}°")
    logger.info(f"    Target roll: -90.0° (X-axis horizontal)")
    logger.info(f"    Roll compensation: {roll_compensation:.2f}° around Z-axis")
    
    # Create roll rotation matrix around Z-axis using Rodrigues' formula
    K_z = np.array([
        [0, -z_axis_world[2], z_axis_world[1]],
        [z_axis_world[2], 0, -z_axis_world[0]],
        [-z_axis_world[1], z_axis_world[0], 0]
    ])
    
    R_roll = (np.eye(3) + 
              np.sin(roll_rotation_angle) * K_z + 
              (1 - np.cos(roll_rotation_angle)) * (K_z @ K_z))
    
    # Apply roll rotation to X-axis to get the new horizontal X-axis
    x_axis_horizontal = R_roll @ x_axis_world
    x_axis_horizontal = x_axis_horizontal / np.linalg.norm(x_axis_horizontal)
    
    logger.info(f"    X-axis after roll correction: [{x_axis_horizontal[0]:.3f}, {x_axis_horizontal[1]:.3f}, {x_axis_horizontal[2]:.3f}]")
    
    # Step 2: Rotate around the now-horizontal X-axis by pitch angle
    # Negate pitch because positive pitch = looking down = need to rotate up
    pitch_rotation_angle = -np.radians(pitch_angle_degrees)
    
    logger.info(f"  Step 2 - Pitch alignment:")
    logger.info(f"    Current pitch: {pitch_angle_degrees:.2f}°")
    logger.info(f"    Pitch compensation: {-pitch_angle_degrees:.2f}° around horizontal X-axis")
    
    # Create pitch rotation matrix around horizontal X-axis
    K_x = np.array([
        [0, -x_axis_horizontal[2], x_axis_horizontal[1]],
        [x_axis_horizontal[2], 0, -x_axis_horizontal[0]],
        [-x_axis_horizontal[1], x_axis_horizontal[0], 0]
    ])
    
    R_pitch = (np.eye(3) + 
               np.sin(pitch_rotation_angle) * K_x + 
               (1 - np.cos(pitch_rotation_angle)) * (K_x @ K_x))
    
    # Combine rotations: first roll, then pitch
    R_combined = R_pitch @ R_roll
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R_combined
    
    # Debug: show what happens to camera's up vector after combined rotation
    cam_up_before = R_c2w @ np.array([0, -1, 0])  # Camera's up (-Y) in world before rotation
    cam_up_after = R_combined @ cam_up_before  # After both rotations
    logger.info(f"  Camera up before alignment: [{cam_up_before[0]:.3f}, {cam_up_before[1]:.3f}, {cam_up_before[2]:.3f}]")
    logger.info(f"  Camera up after alignment:  [{cam_up_after[0]:.3f}, {cam_up_after[1]:.3f}, {cam_up_after[2]:.3f}]")
    
    return transform


# Import metadata utilities from FastVGGT
# These functions handle pitch angle extraction from EXIF and gravity alignment computation
try:
    import sys
    # Add FastVGGT to path if not already there
    fastvggt_path = os.path.join(os.path.dirname(__file__), "../../../../../FastVGGT")
    if os.path.exists(fastvggt_path) and fastvggt_path not in sys.path:
        sys.path.insert(0, os.path.abspath(fastvggt_path))
    
    from metadata_util import (
        get_pitch_angles,
        compute_gravity_alignment_from_pitch_and_poses,
    )
    METADATA_UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import metadata_util from FastVGGT: {e}")
    logger.warning("Gravity alignment will use camera pose analysis only (no EXIF pitch data)")
    METADATA_UTILS_AVAILABLE = False


def _compute_gravity_alignment_from_camera_vectors(
    extrinsics: np.ndarray,
) -> np.ndarray:
    """
    Compute gravity alignment using the first camera's up vector.
    Uses the first camera's pose from the reconstruction model for consistent alignment.
    
    Standard CV/OpenCV camera coordinate system convention:
    - X-axis: right
    - Y-axis: down
    - Z-axis: forward (viewing direction)
    - Up direction: -Y
    
    The alignment targets -Y axis in world space, which after the glTF 
    Y-flip transformation becomes +Y (up) in the final glTF coordinate system.
    
    Args:
        extrinsics: Camera extrinsic matrices (N, 3, 4) or (N, 4, 4) world-to-camera transforms
                    Only the first camera (index 0) is used for alignment.
        
    Returns:
        4x4 transformation matrix to align first camera's up vector with -Y axis
        (which becomes +Y up in glTF after the coordinate system conversion)
    """
    num_cameras = len(extrinsics)
    
    # Ensure we have 4x4 matrices
    if extrinsics.shape[-2:] == (3, 4):
        ext_4x4 = np.zeros((num_cameras, 4, 4))
        ext_4x4[:, :3, :4] = extrinsics
        ext_4x4[:, 3, 3] = 1
    else:
        ext_4x4 = extrinsics
    
    # Use first camera's pose for alignment (consistent with scene alignment)
    cam_to_world_0 = np.linalg.inv(ext_4x4[0])
    
    # Extract up vector from first camera's coordinate system
    # Standard CV/OpenCV: X=right, Y=down, Z=forward (viewing direction)
    # Camera's up direction is -Y (opposite of down)
    up_cam = np.array([0, -1, 0, 0])
    up_world = cam_to_world_0 @ up_cam
    first_cam_up = up_world[:3]
    
    # Debug: log first camera's orientation
    R = cam_to_world_0[:3, :3]
    cam_x = R @ np.array([1, 0, 0])  # X-axis (right) in world
    cam_y = R @ np.array([0, 1, 0])  # Y-axis (down) in world
    cam_z = R @ np.array([0, 0, 1])  # Z-axis (forward) in world
    logger.info(f"  First camera axes in world frame:")
    logger.info(f"    X (right): [{cam_x[0]:.3f}, {cam_x[1]:.3f}, {cam_x[2]:.3f}]")
    logger.info(f"    Y (down):  [{cam_y[0]:.3f}, {cam_y[1]:.3f}, {cam_y[2]:.3f}]")
    logger.info(f"    Z (fwd):   [{cam_z[0]:.3f}, {cam_z[1]:.3f}, {cam_z[2]:.3f}]")
    logger.info(f"    UP (-Y):   [{first_cam_up[0]:.3f}, {first_cam_up[1]:.3f}, {first_cam_up[2]:.3f}]")
    
    # Check which world axis is most vertical
    abs_up = np.abs(first_cam_up)
    most_vertical_axis = np.argmax(abs_up)
    vertical_component = first_cam_up[most_vertical_axis]
    logger.info(f"  Camera up is mostly along world axis {most_vertical_axis} ({'XYZ'[most_vertical_axis]}) with component {vertical_component:.3f}")
    
    # Use first camera's up vector
    up_norm = np.linalg.norm(first_cam_up)
    
    if up_norm < 1e-6:
        logger.warning("First camera up vector is zero. Using identity transform.")
        return np.eye(4)
    
    first_cam_up = first_cam_up / up_norm
    
    # Target: align camera up vector to -Y in world space
    # The glTF transformation later flips Y (M[1,1]=-1), so -Y becomes +Y in glTF (up)
    # Camera UP opposes gravity, so aligning camera UP to -Y makes the final Y+ point up
    target_up = np.array([0, -1, 0])  # Camera UP to -Y, which becomes +Y in glTF after flip
    
    # Compute rotation using Rodrigues' formula
    v = np.cross(first_cam_up, target_up)
    s = np.linalg.norm(v)
    c = np.dot(first_cam_up, target_up)
    
    # Check if already aligned or opposite
    if s < 1e-6:
        if c > 0:  # Already aligned
            logger.info("Cameras already aligned with gravity (Y-axis up after glTF conversion)")
            return np.eye(4)
        else:  # Opposite direction, need 180 degree rotation
            logger.info("Cameras pointing opposite to gravity. Rotating 180 degrees.")
            R = -np.eye(3)
            R[1, 1] = 1  # Keep Y as is since we're flipping around, target is -Y
    else:
        # Skew-symmetric cross-product matrix
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        
        # Rodrigues' rotation formula: R = I + [v]_x + [v]_x^2 * (1-c)/s^2
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
    
    alignment = np.eye(4)
    alignment[:3, :3] = R
    
    # Compute angle of rotation for logging
    rotation_angle = np.arccos(np.clip(c, -1.0, 1.0)) * 180 / np.pi
    
    logger.info(f"Gravity alignment computed from first camera pose")
    logger.info(f"  First camera up vector (world): [{first_cam_up[0]:.3f}, {first_cam_up[1]:.3f}, {first_cam_up[2]:.3f}]")
    logger.info(f"  Target: align to -Y axis (becomes +Y up in glTF after flip)")
    logger.info(f"  Rotation angle to align: {rotation_angle:.2f}°")
    
    return alignment


def set_sky_depth(prediction: Prediction, sky_mask: np.ndarray, sky_depth_def: float = 98.0):
    non_sky_mask = ~sky_mask
    valid_depth = prediction.depth[non_sky_mask]
    if valid_depth.size > 0:
        max_depth = np.percentile(valid_depth, sky_depth_def)
        prediction.depth[sky_mask] = max_depth


def get_conf_thresh(
    prediction: Prediction,
    sky_mask: np.ndarray,
    conf_thresh: float,
    conf_thresh_percentile: float = 10.0,
    ensure_thresh_percentile: float = 90.0,
):
    if sky_mask is not None and (~sky_mask).sum() > 10:
        conf_pixels = prediction.conf[~sky_mask]
    else:
        conf_pixels = prediction.conf
    lower = np.percentile(conf_pixels, conf_thresh_percentile)
    upper = np.percentile(conf_pixels, ensure_thresh_percentile)
    conf_thresh = min(max(conf_thresh, lower), upper)
    return conf_thresh


def export_to_glb(
    prediction: Prediction,
    export_dir: str,
    image_paths: Optional[List[str]] = None,
    num_max_points: int = 1_000_000,
    conf_thresh: float = 1.05,
    filter_black_bg: bool = False,
    filter_white_bg: bool = False,
    use_gravity_alignment: bool = False,
    show_axes: bool = False,
    show_camera_frame: int = -1,
    conf_thresh_percentile: float = 40.0,
    ensure_thresh_percentile: float = 90.0,
    sky_depth_def: float = 98.0,
    show_cameras: bool = True,
    camera_size: float = 0.03,
    export_depth_vis: bool = True,
) -> str:
    """Generate a 3D point cloud and camera wireframes and export them as a ``.glb`` file.

    The function builds a point cloud from the predicted depth maps, aligns it to the
    first camera in glTF coordinates (X-right, Y-up, Z-backward), optionally draws
    camera wireframes, and writes the result to ``scene.glb``. Auxiliary assets such as
    depth visualizations can also be generated alongside the main export.

    Args:
        prediction: Model prediction containing depth, confidence, intrinsics, extrinsics,
            and pre-processed images.
        export_dir: Output directory where the glTF assets will be written.
        image_paths: Optional list of image file paths. Used to extract pitch angles from
            EXIF metadata for more accurate gravity alignment.
        num_max_points: Maximum number of points retained after downsampling.
        conf_thresh: Base confidence threshold used before percentile adjustments.
        filter_black_bg: Mark near-black background pixels for removal during confidence filtering.
        filter_white_bg: Mark near-white background pixels for removal during confidence filtering.
        use_gravity_alignment: If True, align the Y-axis with gravity (vertical up) based on
            sensor metadata (pitch angles from EXIF) and camera poses. If metadata is not available,
            falls back to camera pose analysis only. This ensures the scene is oriented upright.
        show_axes: If True, display coordinate axes (X=red, Y=green, Z=blue) at the origin
            for orientation reference.
        show_camera_frame: If >= 0, visualize the camera frame coordinate system for the specified
            camera index (0=first camera, 1=second, etc.). Shows camera axes in cyan (X/right),
            magenta (Y/down), yellow (Z/forward) at the camera's position.
        conf_thresh_percentile: Lower percentile used when adapting the confidence threshold.
        ensure_thresh_percentile: Upper percentile clamp for the adaptive threshold.
        sky_depth_def: Percentile used to fill sky pixels with plausible depth values.
        show_cameras: Whether to render camera wireframes in the exported scene.
        camera_size: Relative camera wireframe scale as a fraction of the scene diagonal.
        export_depth_vis: Whether to export raster depth visualisations alongside the glTF.

    Returns:
        Path to the exported ``scene.glb`` file.
    """
    # 1) Use prediction.processed_images, which is already processed image data
    assert (
        prediction.processed_images is not None
    ), "Export to GLB: prediction.processed_images is required but not available"
    assert (
        prediction.depth is not None
    ), "Export to GLB: prediction.depth is required but not available"
    assert (
        prediction.intrinsics is not None
    ), "Export to GLB: prediction.intrinsics is required but not available"
    assert (
        prediction.extrinsics is not None
    ), "Export to GLB: prediction.extrinsics is required but not available"
    assert (
        prediction.conf is not None
    ), "Export to GLB: prediction.conf is required but not available"
    logger.info(f"conf_thresh_percentile: {conf_thresh_percentile}")
    logger.info(f"num max points: {num_max_points}")
    logger.info(f"Exporting to GLB with num_max_points: {num_max_points}")
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    # 2) Sky processing (if sky_mask is provided)
    if getattr(prediction, "sky_mask", None) is not None:
        set_sky_depth(prediction, prediction.sky_mask, sky_depth_def)

    # 3) Confidence threshold (if no conf, then no filtering)
    if filter_black_bg:
        prediction.conf[(prediction.processed_images < 16).all(axis=-1)] = 1.0
    if filter_white_bg:
        prediction.conf[(prediction.processed_images >= 240).all(axis=-1)] = 1.0
    conf_thr = get_conf_thresh(
        prediction,
        getattr(prediction, "sky_mask", None),
        conf_thresh,
        conf_thresh_percentile,
        ensure_thresh_percentile,
    )

    # 4) Back-project to world coordinates and get colors (world frame)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        images_u8,
        prediction.conf,
        conf_thr,
    )

    # 5) Based on first camera orientation + glTF axis system, center by point cloud,
    # construct alignment transform, and apply to point cloud
    # If use_gravity_alignment is enabled, compute gravity alignment first
    gravity_transform = np.eye(4)
    if use_gravity_alignment:
        logger.info("Computing gravity alignment from sensor metadata and camera poses...")
        
        # Try to get pitch and roll angles from EXIF metadata
        pitch_angles = None
        roll_angles = None
        
        if image_paths is not None and len(image_paths) > 0:
            # Try to extract pitch and roll from EXIF data in the images
            logger.info("Attempting to extract pitch and roll angles from EXIF metadata...")
            pitch_angles = []
            roll_angles = []
            
            for img_path in image_paths:
                pitch, roll = _extract_pitch_and_roll_from_exif(img_path)
                pitch_angles.append(pitch)
                roll_angles.append(roll)
            
            # Log extraction results
            valid_pitches = [p for p in pitch_angles if p is not None]
            valid_rolls = [r for r in roll_angles if r is not None]
            
            if valid_pitches or valid_rolls:
                logger.info(f"Successfully extracted {len(valid_pitches)}/{len(pitch_angles)} pitch angles")
                logger.info(f"Successfully extracted {len(valid_rolls)}/{len(roll_angles)} roll angles")
                
                # Log first few values for verification
                if valid_pitches:
                    sample_pitches = valid_pitches[:3]
                    logger.info(f"Sample pitch values: {[f'{p:.2f}°' for p in sample_pitches]}")
                if valid_rolls:
                    sample_rolls = valid_rolls[:3]
                    logger.info(f"Sample roll values: {[f'{r:.2f}°' for r in sample_rolls]}")
            else:
                logger.info("No valid pitch or roll angles found in EXIF metadata")
                pitch_angles = None
                roll_angles = None
        
        # Compute gravity alignment using simple two-step rotation (roll + pitch)
        if pitch_angles is not None and roll_angles is not None and len(pitch_angles) > 0:
            # Use first camera's angles for two-step alignment
            first_pitch = pitch_angles[0] if pitch_angles[0] is not None else None
            first_roll = roll_angles[0] if roll_angles[0] is not None else None
            
            if first_pitch is not None and first_roll is not None:
                logger.info(f"Using first camera's pitch angle: {first_pitch:.2f}°, roll angle: {first_roll:.2f}°")
                gravity_transform = _compute_simple_pitch_alignment(
                    pitch_angle_degrees=first_pitch,
                    roll_angle_degrees=first_roll,
                    extrinsics=prediction.extrinsics,
                )
            else:
                logger.info("First camera missing pitch or roll data, skipping gravity alignment")
                gravity_transform = np.eye(4)
        else:
            # Fallback to camera pose-only analysis
            logger.info("No pitch angles available, using camera pose analysis")
            gravity_transform = _compute_gravity_alignment_from_camera_vectors(
                prediction.extrinsics
            )
    
    A = _compute_alignment_transform_first_cam_glTF_center_by_points(
        prediction.extrinsics[0], points, gravity_transform
    )  # (4,4)

    if points.shape[0] > 0:
        points = trimesh.transform_points(points, A)

    # 6) Clean + downsample
    points, colors = _filter_and_downsample(points, colors, num_max_points)

    # 7) Assemble scene (add point cloud first)
    scene = trimesh.Scene()
    if scene.metadata is None:
        scene.metadata = {}
    scene.metadata["hf_alignment"] = A  # For camera wireframes and external reuse

    if points.shape[0] > 0:
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
        scene.add_geometry(pc)

    # 8) Draw cameras (wireframe pyramids), using the same transform A
    if show_cameras and prediction.intrinsics is not None and prediction.extrinsics is not None:
        scene_scale = _estimate_scene_scale(points, fallback=1.0)
        H, W = prediction.depth.shape[1:]
        _add_cameras_to_scene(
            scene=scene,
            K=prediction.intrinsics,
            ext_w2c=prediction.extrinsics,
            image_sizes=[(H, W)] * prediction.depth.shape[0],
            scale=scene_scale * camera_size,
        )

    # 8.5) Draw coordinate axes if requested
    if show_axes:
        scene_scale = _estimate_scene_scale(points, fallback=1.0)
        _add_axes_to_scene(scene, scale=scene_scale * 0.2)
    
    # 8.6) Draw camera frame axes for selected camera if requested
    if show_camera_frame >= 0:
        scene_scale = _estimate_scene_scale(points, fallback=1.0)
        _add_camera_frame_axes(
            scene,
            extrinsics=prediction.extrinsics,
            camera_idx=show_camera_frame,
            transform=A,
            scale=scene_scale * 0.3,  # Slightly larger than world axes
        )

    # 9) Export
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, "scene.glb")
    scene.export(out_path)

    if export_depth_vis:
        export_to_depth_vis(prediction, export_dir)
        os.system(f"cp -r {export_dir}/depth_vis/0000.jpg {export_dir}/scene.jpg")
    return out_path


# =========================
# utilities
# =========================


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def _filter_and_downsample(points: np.ndarray, colors: np.ndarray, num_max: int):
    if points.shape[0] == 0:
        return points, colors
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if points.shape[0] > num_max:
        idx = np.random.choice(points.shape[0], num_max, replace=False)
        points, colors = points[idx], colors[idx]
    return points, colors


def _estimate_scene_scale(points: np.ndarray, fallback: float = 1.0) -> float:
    if points.shape[0] < 2:
        return fallback
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    diag = np.linalg.norm(hi - lo)
    return float(diag if np.isfinite(diag) and diag > 0 else fallback)


def _compute_alignment_transform_first_cam_glTF_center_by_points(
    ext_w2c0: np.ndarray,
    points_world: np.ndarray,
    gravity_transform: np.ndarray = None,
) -> np.ndarray:
    """Computes the transformation matrix to align the scene with glTF standards.

    This function calculates a 4x4 homogeneous matrix that centers the scene's
    point cloud and transforms its coordinate system from the computer vision (CV)
    standard to the glTF standard.

    The transformation process involves these steps:
    1.  **Gravity Alignment (optional)**: If gravity_transform is provided, applies
        it first to align the world Z-axis with gravity (vertical up).
    2.  **Initial Alignment**: Orients the world coordinate system to match the
        first camera's view (x-right, y-down, z-forward).
    3.  **Coordinate System Conversion**: Converts the CV camera frame to the
        glTF frame (x-right, y-up, z-backward) by flipping the Y and Z axes.
    4.  **Centering**: Translates the entire scene so that the median of the
        point cloud becomes the new origin (0,0,0).

    Args:
        ext_w2c0: First camera's world-to-camera extrinsic matrix (3,4) or (4,4)
        points_world: Point cloud in world coordinates (N,3)
        gravity_transform: Optional 4x4 gravity alignment transform

    Returns:
        A 4x4 homogeneous transformation matrix (torch.Tensor or np.ndarray)
        that applies these transformations.  A: X' = A @ [X;1]
    """

    w2c0 = _as_homogeneous44(ext_w2c0).astype(np.float64)

    # CV -> glTF axis transformation
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0  # flip Y
    M[2, 2] = -1.0  # flip Z

    # Build transformation without gravity first
    A_no_center_no_gravity = M @ w2c0
    
    # Apply gravity alignment if provided
    if gravity_transform is not None:
        A_no_center = A_no_center_no_gravity @ gravity_transform
        logger.info(f"Applied gravity alignment to transformation pipeline")
    else:
        A_no_center = A_no_center_no_gravity
    
    # Debug: show what happens to the cardinal axes through the pipeline
    if gravity_transform is not None:
        logger.info(f"Transformation pipeline check:")
        # Start with world axes - in world space, which axis points up?
        # We need to check what the gravity alignment does
        world_y_axis = np.array([0, 1, 0, 1])  # World Y-axis
        world_neg_y_axis = np.array([0, -1, 0, 1])  # World -Y axis
        
        # After gravity alignment: which world axis becomes the "up" direction?
        after_gravity_y = gravity_transform @ world_y_axis
        after_gravity_neg_y = gravity_transform @ world_neg_y_axis
        logger.info(f"  World +Y after gravity: [{after_gravity_y[0]:.3f}, {after_gravity_y[1]:.3f}, {after_gravity_y[2]:.3f}]")
        logger.info(f"  World -Y after gravity: [{after_gravity_neg_y[0]:.3f}, {after_gravity_neg_y[1]:.3f}, {after_gravity_neg_y[2]:.3f}]")
        
        # After w2c and M
        final_y = A_no_center @ world_y_axis
        final_neg_y = A_no_center @ world_neg_y_axis
        logger.info(f"  World +Y after full pipeline: [{final_y[0]:.3f}, {final_y[1]:.3f}, {final_y[2]:.3f}]")
        logger.info(f"  World -Y after full pipeline: [{final_neg_y[0]:.3f}, {final_neg_y[1]:.3f}, {final_neg_y[2]:.3f}]")
        logger.info(f"  -> In glTF, Y-axis should point up [0, +1, 0]")


    # Apply gravity transform to points for centering calculation
    points_for_center = points_world
    if gravity_transform is not None:
        points_for_center = trimesh.transform_points(points_world, gravity_transform)

    # Calculate point cloud center in new coordinate system (use median to resist outliers)
    if points_for_center.shape[0] > 0:
        pts_tmp = trimesh.transform_points(points_for_center, A_no_center)
        center = np.median(pts_tmp, axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)

    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center

    # Combine all transformations
    # Note: gravity_transform is already included in A_no_center if provided
    A = T_center @ A_no_center
    
    return A


def _add_cameras_to_scene(
    scene: trimesh.Scene,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    image_sizes: list[tuple[int, int]],
    scale: float,
) -> None:
    """Draws camera frustums to visualize their position and orientation.

    This function renders each camera as a wireframe pyramid, originating from
    the camera's center and extending to the corners of its imaging plane.

    It reads the 'hf_alignment' metadata from the scene to ensure the
    wireframes are correctly aligned with the 3D point cloud.
    """
    N = K.shape[0]
    if N == 0:
        return

    # Alignment matrix consistent with point cloud (use identity matrix if missing)
    A = None
    try:
        A = scene.metadata.get("hf_alignment", None) if scene.metadata else None
    except Exception:
        A = None
    if A is None:
        A = np.eye(4, dtype=np.float64)

    for i in range(N):
        H, W = image_sizes[i]
        segs = _camera_frustum_lines(K[i], ext_w2c[i], W, H, scale)  # (8,2,3) world frame
        # Apply unified transformation
        segs = trimesh.transform_points(segs.reshape(-1, 3), A).reshape(-1, 2, 3)
        path = trimesh.load_path(segs)
        color = _index_color_rgb(i, N)
        if hasattr(path, "colors"):
            path.colors = np.tile(color, (len(path.entities), 1))
        scene.add_geometry(path)


def _camera_frustum_lines(
    K: np.ndarray, ext_w2c: np.ndarray, W: int, H: int, scale: float
) -> np.ndarray:
    corners = np.array(
        [
            [0, 0, 1.0],
            [W - 1, 0, 1.0],
            [W - 1, H - 1, 1.0],
            [0, H - 1, 1.0],
        ],
        dtype=float,
    )  # (4,3)

    K_inv = np.linalg.inv(K)
    c2w = np.linalg.inv(_as_homogeneous44(ext_w2c))

    # camera center in world
    Cw = (c2w @ np.array([0, 0, 0, 1.0]))[:3]

    # rays -> z=1 plane points (camera frame)
    rays = (K_inv @ corners.T).T
    z = rays[:, 2:3]
    z[z == 0] = 1.0
    plane_cam = (rays / z) * scale  # (4,3)

    # to world
    plane_w = []
    for p in plane_cam:
        pw = (c2w @ np.array([p[0], p[1], p[2], 1.0]))[:3]
        plane_w.append(pw)
    plane_w = np.stack(plane_w, 0)  # (4,3)

    segs = []
    # center to corners
    for k in range(4):
        segs.append(np.stack([Cw, plane_w[k]], 0))
    # rectangle edges
    order = [0, 1, 2, 3, 0]
    for a, b in zip(order[:-1], order[1:]):
        segs.append(np.stack([plane_w[a], plane_w[b]], 0))

    return np.stack(segs, 0)  # (8,2,3)


def _add_axes_to_scene(scene: trimesh.Scene, scale: float = 1.0, origin: np.ndarray = None) -> None:
    """Add coordinate axes to the scene for orientation reference.
    
    These axes show the coordinate system AFTER all transformations (gravity alignment, 
    camera transform, and glTF conversion). They represent the final glTF coordinate system.
    
    Args:
        scene: Trimesh scene to add axes to
        scale: Length of each axis
        origin: Origin point for axes in the transformed coordinate system (default: [0,0,0])
    """
    if origin is None:
        origin = np.zeros(3)
    
    # Create axis lines in the final (transformed) coordinate system
    # These show X (red), Y (green), Z (blue) in glTF space
    # Create each axis separately for proper coloring
    
    # X-axis (red - right)
    x_segments = np.array([[origin, origin + np.array([scale, 0, 0])]])
    x_path = trimesh.load_path(x_segments)
    x_path.colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)
    scene.add_geometry(x_path)
    
    # Y-axis (green - UP)
    y_segments = np.array([[origin, origin + np.array([0, scale, 0])]])
    y_path = trimesh.load_path(y_segments)
    y_path.colors = np.array([[0, 255, 0, 255]], dtype=np.uint8)
    scene.add_geometry(y_path)
    
    # Z-axis (blue - toward camera)
    z_segments = np.array([[origin, origin + np.array([0, 0, scale])]])
    z_path = trimesh.load_path(z_segments)
    z_path.colors = np.array([[0, 0, 255, 255]], dtype=np.uint8)
    scene.add_geometry(z_path)
    
    logger.info(f"Added coordinate axes (scale: {scale:.3f}): X=red, Y=green (UP), Z=blue")


def _add_camera_frame_axes(
    scene: trimesh.Scene,
    extrinsics: np.ndarray,
    camera_idx: int,
    transform: np.ndarray,
    scale: float = 1.0,
) -> None:
    """Add coordinate axes showing the camera's local frame for a specific camera.
    
    This visualizes the camera's coordinate system (X=right, Y=down, Z=forward) 
    in the final transformed coordinate system. Useful for understanding camera orientation.
    
    Args:
        scene: Trimesh scene to add axes to
        extrinsics: Camera extrinsic matrices (N, 3, 4) or (N, 4, 4) - world-to-camera transforms
        camera_idx: Index of the camera to visualize (0-based)
        transform: The full transformation matrix A applied to the scene (includes gravity, w2c, M, center)
        scale: Length of each camera axis
    """
    if camera_idx < 0 or camera_idx >= len(extrinsics):
        logger.warning(f"Invalid camera index {camera_idx}, must be 0-{len(extrinsics)-1}")
        return
    
    # Convert extrinsics to 4x4 if needed
    if extrinsics.shape[-2:] == (3, 4):
        ext_4x4 = np.zeros((len(extrinsics), 4, 4))
        ext_4x4[:, :3, :4] = extrinsics
        ext_4x4[:, 3, 3] = 1
    else:
        ext_4x4 = extrinsics
    
    # Get camera-to-world transform
    w2c = ext_4x4[camera_idx]
    try:
        c2w = np.linalg.inv(w2c)
    except np.linalg.LinAlgError:
        logger.warning(f"Could not invert camera matrix for camera {camera_idx}")
        return
    
    # Camera center in world coordinates
    camera_center_world = (c2w @ np.array([0, 0, 0, 1]))[:3]
    
    # Camera axes in world coordinates
    # In camera frame: X=right, Y=down, Z=forward (viewing direction)
    cam_x_world = (c2w @ np.array([1, 0, 0, 0]))[:3]  # Right
    cam_y_world = (c2w @ np.array([0, 1, 0, 0]))[:3]  # Down
    cam_z_world = (c2w @ np.array([0, 0, 1, 0]))[:3]  # Forward (viewing direction)
    
    # Apply the same transformation that was applied to the point cloud
    # Transform points using the full pipeline (gravity + w2c + M + center)
    camera_center_transformed = trimesh.transform_points(
        camera_center_world.reshape(1, 3), transform
    )[0]
    
    # Transform axis endpoints
    x_end_world = camera_center_world + cam_x_world * scale
    y_end_world = camera_center_world + cam_y_world * scale
    z_end_world = camera_center_world + cam_z_world * scale
    
    x_end_transformed = trimesh.transform_points(x_end_world.reshape(1, 3), transform)[0]
    y_end_transformed = trimesh.transform_points(y_end_world.reshape(1, 3), transform)[0]
    z_end_transformed = trimesh.transform_points(z_end_world.reshape(1, 3), transform)[0]
    
    # Create axis segments - need to create each axis separately for proper coloring
    # X-axis (cyan - right)
    x_segments = np.array([[camera_center_transformed, x_end_transformed]])
    x_path = trimesh.load_path(x_segments)
    x_path.colors = np.array([[0, 255, 255, 255]], dtype=np.uint8)
    scene.add_geometry(x_path)
    
    # Y-axis (magenta - down)
    y_segments = np.array([[camera_center_transformed, y_end_transformed]])
    y_path = trimesh.load_path(y_segments)
    y_path.colors = np.array([[255, 0, 255, 255]], dtype=np.uint8)
    scene.add_geometry(y_path)
    
    # Z-axis (yellow - forward/viewing direction)
    z_segments = np.array([[camera_center_transformed, z_end_transformed]])
    z_path = trimesh.load_path(z_segments)
    z_path.colors = np.array([[255, 255, 0, 255]], dtype=np.uint8)
    scene.add_geometry(z_path)
    
    logger.info(f"Added camera #{camera_idx} frame axes (scale: {scale:.3f}): "
                f"X=cyan (right), Y=magenta (down), Z=yellow (forward)")


def _index_color_rgb(i: int, n: int) -> np.ndarray:
    h = (i + 0.5) / max(n, 1)
    s, v = 0.85, 0.95
    r, g, b = _hsv_to_rgb(h, s, v)
    return (np.array([r, g, b]) * 255).astype(np.uint8)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return r, g, b
