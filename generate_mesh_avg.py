#!/usr/bin/env python3
"""
Enhanced Mesh Distance Comparison Tool

This script compares meshes from NIfTI images, supports batch processing,
and visualizes average distances on the atlas mesh to show alignment quality.
USAGE ex: python3 generate_mesh_avg.py Atlas.nii.gz -d <dir>
"""

import numpy as np
import nibabel as nib
from skimage import measure
from skimage.filters import threshold_otsu
import open3d as o3d
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fast_simplification
import os
import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import json

# Default parameters
DEFAULT_SIMPLIFICATION_THRESHOLD = 400000


class MeshProcessor:
    """Class to handle mesh processing operations."""
    
    def __init__(self, simplification_threshold=DEFAULT_SIMPLIFICATION_THRESHOLD):
        self.simplification_threshold = simplification_threshold
    
    def load_and_preprocess_nifti(self, filepath: str, threshold: Optional[float] = None) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Load NIfTI file and preprocess for mesh generation."""
        print(f"Loading {os.path.basename(filepath)}...")
        img = nib.load(filepath)
        
        # Get canonical orientation for consistent processing
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()
        
        # Get voxel sizes from affine
        voxel_sizes = nib.affines.voxel_sizes(img.affine)
        
        # Calculate threshold if not provided
        if threshold is None:
            threshold = self.calculate_auto_threshold(data)
            print(f"  Auto-calculated threshold: {threshold:.3f}")
        
        # Print data statistics
        print(f"  Shape: {data.shape}, Range: [{np.min(data):.1f}, {np.max(data):.1f}]")
        print(f"  Voxel sizes: {voxel_sizes}")
        
        return data, threshold, voxel_sizes, img.affine
    
    @staticmethod
    def calculate_auto_threshold(data: np.ndarray) -> float:
        """Calculate threshold using Otsu's method."""
        flat_data = data.flatten()
        flat_data = flat_data[flat_data > 0]
        return threshold_otsu(flat_data) if len(flat_data) > 0 else 0.075
    
    def generate_mesh_marching_cubes(self, data: np.ndarray, threshold: float, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print(f"  Generating mesh (threshold={threshold:.3f})...")
        # 1) Extract in IJK index space
        verts_ijk, faces, _, _ = measure.marching_cubes(
            data,
            level=threshold,
            spacing=(1.0, 1.0, 1.0)
        )
        # 2) Map to world coordinates with affine
        verts_h = np.c_[verts_ijk, np.ones(len(verts_ijk))]
        verts_world = (affine @ verts_h.T).T[:, :3]
        print(f"  Generated: {len(verts_world):,} vertices, {len(faces):,} faces")
        return verts_world, faces
    
    def simplify_mesh_if_needed(self, vertices: np.ndarray, faces: np.ndarray, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Simplify mesh if it exceeds threshold."""
        if force or len(vertices) > self.simplification_threshold:
            factor = max(min(1, 1 - (self.simplification_threshold / len(vertices))), 0)
            print(f"  Simplifying mesh (factor={factor:.3f})...")
            vertices, faces = fast_simplification.simplify(vertices, faces, factor)
            print(f"  Simplified: {len(vertices):,} vertices, {len(faces):,} faces")
        return vertices, faces
    
    @staticmethod
    def calculate_distances_from_atlas_to_specimen(atlas_vertices: np.ndarray, 
                                                   specimen_vertices: np.ndarray, 
                                                   specimen_faces: np.ndarray) -> np.ndarray:
        """
        Calculate distances from atlas vertices to specimen mesh surface.
        This ensures consistent vertex correspondence for averaging.
        """
        print("  Calculating distances from atlas vertices to specimen surface...")
        
        # Create specimen mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(specimen_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(specimen_faces)
        
        # Create scene for distance computation
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        
        # Compute closest points from atlas vertices to specimen surface
        points = o3d.core.Tensor(atlas_vertices.astype(np.float32))
        result = scene.compute_closest_points(points)
        closest_points = result['points'].numpy()
        
        # Calculate distances
        distances = np.linalg.norm(atlas_vertices - closest_points, axis=1)
        
        return distances


class BatchProcessor:
    """Class to handle batch processing of multiple volumes."""
    
    def __init__(self, mesh_processor: MeshProcessor):
        self.mesh_processor = mesh_processor
    
    def process_volume(self, volume_path: str, threshold: Optional[float] = None, 
                      simplify: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single volume and return mesh data."""
        print(f"\nProcessing: {os.path.basename(volume_path)}")
        
        # Loads and processes volume
        data, used_threshold, voxel_sizes, affine = self.mesh_processor.load_and_preprocess_nifti(volume_path, threshold)
        
        # Generates mesh with proper world coordinates
        vertices, faces = self.mesh_processor.generate_mesh_marching_cubes(data, used_threshold, affine)
        
        # Optionally simplify
        if simplify:
            vertices, faces = self.mesh_processor.simplify_mesh_if_needed(vertices, faces)
        
        return vertices, faces
    
    def process_directory(self, directory: str, pattern: str = "*.nii.gz", 
                         threshold: Optional[float] = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Process all volumes in a directory."""
        volume_paths = sorted(glob.glob(os.path.join(directory, pattern)))
        
        if not volume_paths:
            raise ValueError(f"No files matching pattern '{pattern}' found in {directory}")
        
        print(f"Found {len(volume_paths)} volumes to process")
        
        results = []
        for path in volume_paths:
            try:
                vertices, faces = self.process_volume(path, threshold)
                results.append((path, vertices, faces))
            except Exception as e:
                print(f"  Error processing {os.path.basename(path)}: {e}")
                continue
        
        return results
    
    def calculate_average_distances_on_atlas(self, atlas_vertices: np.ndarray, atlas_faces: np.ndarray,
                                            specimen_results: List[Tuple[str, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, List[np.ndarray], dict]:
        """
        Calculate average distances sampled on atlas vertices.
        This ensures proper spatial correspondence for averaging.
        """
        print("\n### Computing Distances from Atlas to Each Specimen ###")
        
        per_specimen_distances = []
        specimen_stats = {}
        
        for path, specimen_vertices, specimen_faces in specimen_results:
            specimen_name = os.path.basename(path)
            print(f"  Processing {specimen_name}...")
            
            # Calculate distances from atlas vertices to this specimen
            distances = self.mesh_processor.calculate_distances_from_atlas_to_specimen(
                atlas_vertices, specimen_vertices, specimen_faces
            )
            
            per_specimen_distances.append(distances)
            
            # Calculate per-specimen statistics
            specimen_stats[specimen_name] = {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'max': float(np.max(distances)),
                'percentile_95': float(np.percentile(distances, 95))
            }
            
            print(f"    Mean: {specimen_stats[specimen_name]['mean']:.3f}mm, "
                  f"Max: {specimen_stats[specimen_name]['max']:.3f}mm")
        
        # Calculate average distances across all specimens
        if per_specimen_distances:
            # Stack all distance arrays (all have same length = len(atlas_vertices))
            distances_array = np.vstack(per_specimen_distances)
            
            # Calculate statistics
            avg_distances = np.mean(distances_array, axis=0)
            std_distances = np.std(distances_array, axis=0)
            
            print(f"\n  Overall Average Distance: {np.mean(avg_distances):.3f}mm ± {np.mean(std_distances):.3f}mm")
            
            return avg_distances, per_specimen_distances, specimen_stats
        
        return np.array([]), [], {}


class Visualizer:
    """Class to handle visualization."""
    
    def __init__(self):
        self.range_modes = [
            (3.0, "3 STD"),
            (2.0, "2 STD"), 
            (1.0, "1 STD"),
            (10.0, "Fixed 10mm"),
            (5.0, "Fixed 5mm"),
            (3.0, "Fixed 3mm"),
            (1.0, "Fixed 1mm"),
            (0.5, "Fixed 0.5mm")
        ]
        self.current_mode_idx = 0
    
    def visualize_with_stats(self, vertices: np.ndarray, faces: np.ndarray, 
                             distances: np.ndarray, title: str = "Mesh Distance Heatmap",
                             specimen_count: int = 1, specimen_stats: dict = None):
        """Interactive VTK visualization with statistics display."""
        print(f"\nCreating visualization: {title}")
        
        # Calculate statistics
        mean_dist = np.mean(distances) if len(distances) > 0 else 0
        std_dist = np.std(distances) if len(distances) > 0 else 0.1
        max_dist = np.max(distances) if len(distances) > 0 else 0
        
        # Create VTK components
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        
        cells = vtk.vtkCellArray()
        for face in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, face[0])
            triangle.GetPointIds().SetId(1, face[1])
            triangle.GetPointIds().SetId(2, face[2])
            cells.InsertNextCell(triangle)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)
        
        # Setup visualization components
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.1)
        
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetWindowName(title)
        render_window.SetSize(1200, 800)
        
        # Create text displays
        self._create_text_actors(renderer, mean_dist, std_dist, max_dist, specimen_count)
        
        # Create color bar
        scalar_bar = self._create_scalar_bar(renderer)
        
        # Update function
        def update_visualization():
            max_distance = self._get_max_distance(self.current_mode_idx, std_dist)
            
            # Set distances as scalars (they match vertex count exactly now)
            vtk_distances = numpy_support.numpy_to_vtk(distances)
            vtk_distances.SetName("Distances")
            polydata.GetPointData().SetScalars(vtk_distances)
            
            # Configure mapper
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToMapScalars()
            mapper.SetScalarRange(0, max_distance)
            
            # Create lookup table
            lut = self._create_lookup_table(max_distance)
            mapper.SetLookupTable(lut)
            scalar_bar.SetLookupTable(lut)
            
            # Update range text
            self.range_text.SetInput(f"Range: 0 to {max_distance:.3f}mm ({self.range_modes[self.current_mode_idx][1]})")
            
            render_window.Render()
        
        # Key press callback
        def key_press_callback(obj, event):
            key = obj.GetKeySym()
            if key == "Tab":
                self.current_mode_idx = (self.current_mode_idx + 1) % len(self.range_modes)
                update_visualization()
                print(f"Switched to: {self.range_modes[self.current_mode_idx][1]}")
            elif key == "s" or key == "S":
                self._save_statistics(distances, specimen_count, specimen_stats)
        
        # Setup interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.AddObserver("KeyPressEvent", key_press_callback)
        
        # Setup camera
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        camera.Zoom(1.2)
        
        # Initial update
        update_visualization()
        
        # Start interaction
        render_window.Render()
        interactor.Start()
    
    def _create_text_actors(self, renderer, mean_dist, std_dist, max_dist, specimen_count):
        """Create text actors for display."""
        # Range text
        self.range_text = vtk.vtkTextActor()
        self.range_text.SetPosition(10, 750)
        self.range_text.GetTextProperty().SetFontSize(16)
        self.range_text.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        renderer.AddActor2D(self.range_text)
        
        # Statistics text
        stats_text = vtk.vtkTextActor()
        stats_text.SetInput(f"Mean: {mean_dist:.3f}mm | STD: {std_dist:.3f}mm | Max: {max_dist:.3f}mm | Specimens: {specimen_count}")
        stats_text.SetPosition(10, 720)
        stats_text.GetTextProperty().SetFontSize(14)
        stats_text.GetTextProperty().SetColor(0.9, 0.9, 0.9)
        renderer.AddActor2D(stats_text)
        
        # Instructions
        instruction_text = vtk.vtkTextActor()
        instruction_text.SetInput("TAB: Cycle ranges | S: Save statistics | Mouse: Rotate/Zoom")
        instruction_text.SetPosition(10, 690)
        instruction_text.GetTextProperty().SetFontSize(12)
        instruction_text.GetTextProperty().SetColor(0.7, 0.7, 0.7)
        renderer.AddActor2D(instruction_text)
    
    def _create_scalar_bar(self, renderer):
        """Create scalar bar for color mapping."""
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetTitle("Distance (mm)")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.85, 0.1)
        scalar_bar.SetWidth(0.1)
        scalar_bar.SetHeight(0.8)
        scalar_bar.GetTitleTextProperty().SetColor(1.0, 1.0, 1.0)
        scalar_bar.GetLabelTextProperty().SetColor(1.0, 1.0, 1.0)
        renderer.AddActor2D(scalar_bar)
        return scalar_bar
    
    def _create_lookup_table(self, max_distance):
        """Create color lookup table."""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        colormap = cm.get_cmap('turbo')
        for i in range(256):
            rgba = colormap(i / 255.0)
            lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], 1.0)
        lut.SetRange(0, max_distance)
        lut.Build()
        return lut
    
    def _get_max_distance(self, mode_idx, std_dist):
        """Get maximum distance for current mode."""
        multiplier, desc = self.range_modes[mode_idx]
        if "Fixed" in desc:
            return multiplier
        else:
            return multiplier * std_dist
    
    def _save_statistics(self, distances, specimen_count, specimen_stats=None):
        """Save statistics to file."""
        stats = {
            'specimen_count': specimen_count,
            'overall_statistics': {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'max_distance': float(np.max(distances)),
                'min_distance': float(np.min(distances)),
                'percentiles': {
                    '25': float(np.percentile(distances, 25)),
                    '50': float(np.percentile(distances, 50)),
                    '75': float(np.percentile(distances, 75)),
                    '95': float(np.percentile(distances, 95))
                }
            }
        }
        
        if specimen_stats:
            stats['per_specimen'] = specimen_stats
        
        filename = f"distance_stats_{specimen_count}_specimens.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {filename}")
    
    def plot_distance_histogram(self, distances_list: List[np.ndarray], labels: List[str] = None,
                                title_suffix: str = ""):
        """Plot histogram of distances for multiple specimens."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Individual histograms
        for i, distances in enumerate(distances_list):
            label = labels[i] if labels else f"Specimen {i+1}"
            ax1.hist(distances, bins=50, alpha=0.5, label=label[:30])  # Truncate long labels
        
        ax1.set_xlabel("Distance (mm)")
        ax1.set_ylabel("Vertex Count")
        ax1.set_title("Distance Distribution by Specimen")
        if len(distances_list) <= 10:  # Only show legend if not too many specimens
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Combined/average histogram
        all_distances = np.concatenate(distances_list)
        ax2.hist(all_distances, bins=50, color='blue', alpha=0.7)
        ax2.axvline(np.mean(all_distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_distances):.3f}mm')
        ax2.axvline(np.median(all_distances), color='green', linestyle='--', 
                   label=f'Median: {np.median(all_distances):.3f}mm')
        
        ax2.set_xlabel("Distance (mm)")
        ax2.set_ylabel("Vertex Count")
        ax2.set_title(f"Combined Distance Distribution{title_suffix}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("distance_distribution.png", dpi=150)
        plt.show()


def create_dummy_data():
    """Create dummy NIfTI files for testing."""
    os.makedirs("./test_data", exist_ok=True)
    
    size = 200
    
    # Create atlas (sphere)
    atlas_data = np.zeros((size, size, size))
    center = size // 2
    radius = 70
    x, y, z = np.ogrid[:size, :size, :size]
    mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
    atlas_data[mask] = 1000
    
    atlas_nifti = nib.Nifti1Image(atlas_data, np.eye(4))
    nib.save(atlas_nifti, "./test_data/atlas.nii.gz")
    
    # Create multiple test specimens with variations
    for i in range(3):
        specimen_data = np.zeros((size, size, size))
        
        # Add some variation
        offset = np.random.randint(-10, 10, 3)
        radius_var = radius + np.random.randint(-5, 5)
        
        x, y, z = np.ogrid[:size, :size, :size]
        mask = ((x - center - offset[0])**2 + 
                (y - center - offset[1])**2 + 
                (z - center - offset[2])**2 <= radius_var**2)
        specimen_data[mask] = 1000
        
        specimen_nifti = nib.Nifti1Image(specimen_data, np.eye(4))
        nib.save(specimen_nifti, f"./test_data/specimen_{i+1}.nii.gz")
    
    print("Created test data in ./test_data/")
    return "./test_data/atlas.nii.gz", "./test_data/"


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Mesh Distance Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare single specimen to atlas
  %(prog)s atlas.nii.gz specimen.nii.gz
  
  # Process directory of specimens
  %(prog)s atlas.nii.gz --directory /path/to/specimens/
  
  # Use specific thresholds
  %(prog)s atlas.nii.gz specimen.nii.gz --threshold-atlas 0.5 --threshold-specimen 0.6
  
  # Test with dummy data
  %(prog)s --test
        """
    )
    
    parser.add_argument("atlas", nargs="?", help="Path to atlas NIfTI file")
    parser.add_argument("specimen", nargs="?", help="Path to specimen NIfTI file (or use --directory)")
    
    parser.add_argument("-d", "--directory", help="Directory containing specimen NIfTI files")
    parser.add_argument("-p", "--pattern", default="*.nii.gz", help="File pattern for directory processing (default: *.nii.gz)")
    
    parser.add_argument("--threshold-atlas", type=float, help="Threshold for atlas (auto if not specified)")
    parser.add_argument("--threshold-specimen", type=float, help="Threshold for specimens (auto if not specified)")
    
    parser.add_argument("--no-simplification", action="store_true", help="Disable mesh simplification")
    parser.add_argument("--simplification-threshold", type=int, default=DEFAULT_SIMPLIFICATION_THRESHOLD,
                       help=f"Vertex count threshold for simplification (default: {DEFAULT_SIMPLIFICATION_THRESHOLD})")
    
    parser.add_argument("--save-meshes", action="store_true", help="Save processed meshes to files")
    parser.add_argument("--output-dir", default="./output", help="Output directory for saved files")
    
    parser.add_argument("--test", action="store_true", help="Run with test data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        print("Running in test mode...")
        atlas_path, specimen_dir = create_dummy_data()
        args.atlas = atlas_path
        args.directory = specimen_dir
        args.threshold_atlas = 500
        args.threshold_specimen = 500
    
    # Validate arguments
    if not args.atlas:
        parser.error("Atlas file is required (or use --test)")
    
    if not args.specimen and not args.directory:
        parser.error("Either specimen file or --directory must be specified")
    
    # Create output directory if needed
    if args.save_meshes:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processors
    mesh_processor = MeshProcessor(
        simplification_threshold=args.simplification_threshold if not args.no_simplification else float('inf')
    )
    
    batch_processor = BatchProcessor(mesh_processor)
    visualizer = Visualizer()
    
    print("="*60)
    print("ENHANCED MESH DISTANCE COMPARISON TOOL")
    print("="*60)
    
    # Process atlas (don't simplify atlas to maintain vertex correspondence)
    print("\n### Processing Atlas ###")
    atlas_vertices, atlas_faces = batch_processor.process_volume(
        args.atlas, 
        args.threshold_atlas,
        simplify=False  # Keep atlas at full resolution for accurate sampling
    )
    
    # Process specimens
    specimen_results = []
    
    if args.directory:
        print(f"\n### Processing Directory: {args.directory} ###")
        specimen_results = batch_processor.process_directory(
            args.directory, args.pattern, args.threshold_specimen
        )
    else:
        print("\n### Processing Single Specimen ###")
        vertices, faces = batch_processor.process_volume(args.specimen, args.threshold_specimen)
        specimen_results = [(args.specimen, vertices, faces)]
    
    if not specimen_results:
        print("No specimens were successfully processed!")
        return
    
    # Calculate distances
    if len(specimen_results) > 1:
        # Calculate average distances on atlas vertices
        avg_distances, all_distances, specimen_stats = batch_processor.calculate_average_distances_on_atlas(
            atlas_vertices, atlas_faces, specimen_results
        )
        
        print("\n### Visualization ###")
        print("Showing average distance heatmap on ATLAS mesh")
        print("Blue/Green = Good alignment | Yellow/Orange = Moderate | Red = Poor alignment")
        
        # Visualize on atlas mesh
        visualizer.visualize_with_stats(
            atlas_vertices, atlas_faces, avg_distances,
            title=f"Atlas: Average Distance from {len(specimen_results)} Specimens",
            specimen_count=len(specimen_results),
            specimen_stats=specimen_stats
        )
        
        # Plot histograms
        labels = [os.path.basename(path) for path, _, _ in specimen_results]
        visualizer.plot_distance_histogram(all_distances, labels, 
                                          f" ({len(specimen_results)} specimens)")
        
    else:
        # Single specimen comparison
        print("\n### Computing Distances ###")
        path, specimen_vertices, specimen_faces = specimen_results[0]
        
        # Calculate distances from atlas to specimen
        distances = mesh_processor.calculate_distances_from_atlas_to_specimen(
            atlas_vertices, specimen_vertices, specimen_faces
        )
        
        print(f"\nDistance Statistics:")
        print(f"  Mean: {np.mean(distances):.3f} mm")
        print(f"  STD:  {np.std(distances):.3f} mm")
        print(f"  Max:  {np.max(distances):.3f} mm")
        print(f"  95th percentile: {np.percentile(distances, 95):.3f} mm")
        
        # Visualize on atlas mesh
        visualizer.visualize_with_stats(
            atlas_vertices, atlas_faces, distances,
            title=f"Atlas: Distance to {os.path.basename(path)}",
            specimen_count=1
        )
    
    # Save meshes if requested
    if args.save_meshes:
        print(f"\n### Saving Meshes to {args.output_dir} ###")
        
        # Save atlas mesh
        atlas_mesh = o3d.geometry.TriangleMesh()
        atlas_mesh.vertices = o3d.utility.Vector3dVector(atlas_vertices)
        atlas_mesh.triangles = o3d.utility.Vector3iVector(atlas_faces)
        atlas_filename = os.path.join(args.output_dir, "atlas_mesh.ply")
        o3d.io.write_triangle_mesh(atlas_filename, atlas_mesh)
        print(f"  Saved: {atlas_filename}")
        
        # Save specimen meshes
        for i, (path, vertices, faces) in enumerate(specimen_results):
            specimen_mesh = o3d.geometry.TriangleMesh()
            specimen_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            specimen_mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            base_name = os.path.splitext(os.path.basename(path))[0]
            specimen_filename = os.path.join(args.output_dir, f"{base_name}_mesh.ply")
            o3d.io.write_triangle_mesh(specimen_filename, specimen_mesh)
            print(f"  Saved: {specimen_filename}")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("Visualization shows which ATLAS regions align better/worse across specimens")
    print("="*60)


if __name__ == "__main__":
    main()
