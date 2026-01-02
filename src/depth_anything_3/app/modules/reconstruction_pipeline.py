import os
import torch
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List, Dict, Any
from depth_anything_3.app.modules.model_inference import ModelInference
from depth_anything_3.utils.export.glb import (
    export_to_glb, 
    _depths_to_world_points_with_colors, 
    get_conf_thresh,
    _extract_pitch_and_roll_from_exif,
    _extract_pitch_and_roll_from_exif,
    get_conf_thresh,
    _compute_simple_pitch_alignment,
    _compute_gravity_alignment_from_camera_vectors,
    _compute_alignment_transform_first_cam_glTF_center_by_points
)
import trimesh
from scipy.spatial.transform import Rotation

class ReconstructionPipeline:
    """
    Pipeline for 3D reconstruction using Depth Anything 3.
    """
    def __init__(self, device: str = "cuda"):
        self.model_inference = ModelInference()
        self.device = device
        # Initialize model immediately
        self.model_inference.initialize_model(device=self.device)

    def run_reconstruction(
        self,
        target_dir: str,
        image_paths: List[str],
        save_ply: bool = True,
        save_glb: bool = True,
        num_max_points: int = 1_000_000,
        save_percentage: float = 30.0,
        use_gravity_alignment: bool = True,
        use_z_up: bool = True,
    ) -> Dict[str, str]:
        """
        Run the reconstruction pipeline.

        Args:
            target_dir: Directory where results will be saved.
            image_paths: List of paths to input images.
            save_ply: Whether to save the point cloud as .ply (required for SpatialLM).
            save_glb: Whether to save the scene as .glb (for visualization).
            num_max_points: Maximum number of points in the point cloud.
            save_percentage: Confidence threshold percentile.

        Returns:
            Dictionary containing paths to generated files.
        """
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Run inference
        # ModelInference.run_inference expects images in target_dir/images usually, 
        # but we can pass image_paths directly to model.inference if we bypassed run_inference.
        # However, run_inference handles a lot of setup. 
        # Let's look at ModelInference.run_inference again. It finds images in target_dir/images.
        # So we should probably link or copy images there if they aren't already.
        
        # But wait, the server will save images to target_dir/images.
        # So we can just call run_inference.
        
        print(f"Running reconstruction in {target_dir}")
        
        # We need to make sure images are in target_dir/images
        images_dir = os.path.join(target_dir, "images")
        if not os.path.exists(images_dir):
             # If images are not in the expected structure, we might need to handle it.
             # But for now, let's assume the server sets it up correctly.
             pass

        with torch.no_grad():
            prediction, processed_data = self.model_inference.run_inference(
                target_dir=target_dir,
                process_res_method="upper_bound_resize", # Default low_res
                save_percentage=save_percentage,
                num_max_points=num_max_points,
                infer_gs=False, # We don't need GS for SpatialLM
                use_gravity_alignment=use_gravity_alignment,
                use_z_up=use_z_up,
            )

        results = {}
        
        if save_glb:
            # run_inference already calls export_to_glb and saves to scene.glb
            results["glb"] = os.path.join(target_dir, "scene.glb")

        if save_ply:
            if "glb" in results and os.path.exists(results["glb"]):
                # Create PLY by converting GLB to ensure consistency
                print("Converting GLB to PLY for consistency...")
                ply_path = os.path.join(target_dir, "scene.ply")
                
                try:
                    # Load GLB using trimesh
                    scene = trimesh.load(results["glb"], force='scene')
                    
                    # Merge scene geometry into a single point cloud or mesh
                    # If it's a point cloud GLB, scene.geometry will contain PointCloud objects
                    
                    # Collect all points and colors
                    all_points = []
                    all_colors = []
                    
                    for geom in scene.geometry.values():
                        if hasattr(geom, 'vertices'):
                            points = geom.vertices
                            colors = None
                            if hasattr(geom, 'visual') and hasattr(geom.visual, 'vertex_colors'):
                                colors = geom.visual.vertex_colors
                            elif hasattr(geom, 'visual') and hasattr(geom.visual, 'face_colors'):
                                # Fallback? GLB export from depth anything is likely points
                                pass
                                
                            all_points.append(points)
                            if colors is not None:
                                all_colors.append(colors)
                                
                    if all_points:
                        combined_points = np.vstack(all_points)
                        combined_colors = np.vstack(all_colors) if all_colors else None
                        
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(combined_points)
                        if combined_colors is not None:
                             # Trimesh colors are typically uint8 (0-255)
                             # Open3D expects float (0-1)
                             pcd.colors = o3d.utility.Vector3dVector(combined_colors[:, :3] / 255.0)
                        
                        o3d.io.write_point_cloud(ply_path, pcd)
                        results["ply"] = ply_path
                        print(f"Saved PLY to {ply_path}")
                    else:
                        print("Warning: No geometry found in GLB to convert to PLY")
                except Exception as e:
                    print(f"Error converting GLB to PLY: {e}")
            else:
                print("Cannot save PLY: GLB file missing (export to GLB required for PLY generation)")

        return results
