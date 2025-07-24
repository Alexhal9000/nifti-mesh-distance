#!/usr/bin/env python3
"""
Mesh Distance Comparison Tool

This script takes two NIfTI images, generates meshes using marching cubes,
extracts their outer shells, computes vertex distances between them, and
displays the first mesh with a color-coded distance heatmap in VTK.
"""

import numpy as np
import nibabel as nib
from skimage import measure
import open3d as o3d
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fast_simplification
import os

# Hardcoded parameters
SCAN1_PATH = os.path.expanduser("~/Documents/MouseData/Eva_Embryos_17_5/extracted/Z_Ctr_29_Eby_1/Z_Ctr_29_Eby_1_edit_3_cropped.nii.gz")
SCAN2_PATH = os.path.expanduser("~/Documents/MouseData/Eva_Embryos_17_5/extracted/Z_Ctr_29_Eby_3/Z_Ctr_29_Eby_3_edit_4_elastic.nii.gz")
THRESHOLD1 = 7628  # Threshold for scan 1
THRESHOLD2 = 6129  # Threshold for scan 2

# for debugging create a dummy nii.gz file of a cube and one of a sphere contained in the cube
def create_dummy_data():
    """Create dummy NIfTI files for testing: a cube and a sphere."""
    # Create dummy_data directory if it doesn't exist
    os.makedirs("./dummy_data", exist_ok=True)
    
    # Create a 300x300x300 volume
    size = 300
    
    # Create cube data (filled cube in center)
    cube_data = np.zeros((size, size, size))
    center = size // 2
    cube_size = 200
    start = center - cube_size // 2
    end = center + cube_size // 2
    cube_data[start:end, start:end, start:end] = 10000
    
    # Create sphere data (filled sphere in center)
    sphere_data = np.zeros((size, size, size))
    radius = 100
    x, y, z = np.ogrid[:size, :size, :size]
    mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
    sphere_data[mask] = 10000
    
    # Save as NIfTI files
    cube_nifti = nib.Nifti1Image(cube_data, np.eye(4))
    sphere_nifti = nib.Nifti1Image(sphere_data, np.eye(4))
    
    nib.save(cube_nifti, "./dummy_data/cube.nii.gz")
    nib.save(sphere_nifti, "./dummy_data/sphere.nii.gz")
    
    print("Created dummy data files: cube.nii.gz and sphere.nii.gz")

# Ask user which data to use
print("=== Mesh Distance Comparison Tool ===")
print("Choose data source:")
print("1. Test with dummy data (cube and sphere)")
print("2. Use provided data files")

while True:
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice in ['1', '2']:
        break
    print("Please enter 1 or 2")

if choice == '1':
    # Create dummy data if files don't exist
    if not os.path.exists("./dummy_data/cube.nii.gz") or not os.path.exists("./dummy_data/sphere.nii.gz"):
        create_dummy_data()
    
    SCAN1_PATH = "./dummy_data/cube.nii.gz"
    SCAN2_PATH = "./dummy_data/sphere.nii.gz"
    THRESHOLD1 = 5000
    THRESHOLD2 = 5000
    print("Using dummy data (cube and sphere)")
else:
    # Use the original hardcoded paths and thresholds
    print("Using provided data files")
    print(f"Scan 1: {SCAN1_PATH}")
    print(f"Scan 2: {SCAN2_PATH}")

VOXEL_SIZE = (1.0, 1.0, 1.0)  # Voxel spacing
QUICK_SIMPLIFICATION = True


def load_and_preprocess_nifti(filepath, threshold, voxel_size):
    """Load NIfTI file and preprocess for mesh generation."""
    print(f"Loading {filepath}...")
    nifti_img = nib.load(filepath)
    nifti_data = nifti_img.get_fdata()
    
    # Swap x and z axes to match BabylonJS coordinate system
    nifti_data = np.swapaxes(nifti_data, 0, 2)
    
    print(f"Data shape: {nifti_data.shape}, threshold: {threshold}")
    return nifti_data


def generate_mesh_marching_cubes(data, threshold, voxel_size):
    """Generate mesh using marching cubes."""
    print(f"Generating mesh with marching cubes at threshold {threshold}...")
    vertices, faces, _, _ = measure.marching_cubes(
        data, 
        level=threshold, 
        spacing=voxel_size
    )
    
    # Invert face normals if needed
    faces = faces[:, ::-1]
    
    print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")

    return vertices, faces


def get_outer_mesh_with_mapping(vertices, faces):
    """
    Extract outer mesh while tracking which original vertices are retained.
    Returns: outer_vertices, outer_faces, vertex_mapping
    """
    print("Extracting outer shell...")
    
    # Create the original mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_triangle_normals()
    
    # Extract triangle data
    normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    
    # Compute triangle centers
    centers = np.mean(verts[triangles], axis=1)
    
    # Define a small offset to avoid self-intersection
    epsilon = 1e-3
    ray_origins = centers + epsilon * normals
    ray_directions = normals
    
    # Create rays
    rays = np.hstack((ray_origins, ray_directions)).astype(np.float32)
    
    # Set up ray casting scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    
    # Cast all rays at once
    result = scene.cast_rays(rays)
    hit_distances = result['t_hit'].numpy()
    
    # Outer triangles have rays that don't hit anything (t_hit is inf)
    outer_indices = np.where(np.isinf(hit_distances))[0]
    outer_faces = faces[outer_indices]
    
    # Find unique vertices referenced by outer faces
    unique_vertex_indices = np.unique(outer_faces.flatten())
    
    # Create mapping from new indices to original indices
    vertex_mapping = unique_vertex_indices
    
    # Create new vertex array with only the vertices used by outer faces
    outer_vertices = vertices[unique_vertex_indices]
    
    # Remap face indices to use the new vertex array
    inverse_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_vertex_indices)}
    remapped_outer_faces = np.array([[inverse_mapping[orig_idx] for orig_idx in face] for face in outer_faces])
    
    print(f"Outer shell: {len(outer_vertices)} vertices, {len(remapped_outer_faces)} faces")
    return outer_vertices, remapped_outer_faces, vertex_mapping


def calculate_landmark_distances(landmarks, vertices, faces):
    """
    Calculate the shortest distance between each landmark and the mesh surface.
    """
    print("Calculating distances to mesh surface...")
    
    # Create an Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Create a Scene with the mesh for distance computation
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh_t)

    # Convert landmarks to tensor
    points = o3d.core.Tensor(landmarks.astype(np.float32))
    
    # Compute closest points and distances
    result = scene.compute_closest_points(points)
    closest_points = result['points'].numpy()
    
    # Calculate distances
    distances = np.linalg.norm(landmarks - closest_points, axis=1)

    return distances


def create_color_map_with_range(distances, max_distance_value):
    """Create color map from distances using rainbow scale with specific max distance."""
    # Calculate statistics for reference
    mean_dist = np.mean(distances[distances > 0])
    std_dist = np.std(distances[distances > 0])
    
    # Normalize distances to 0-1 range based on provided max distance
    normalized_distances = np.clip(distances / max_distance_value, 0, 1)
    
    # Create rainbow colormap (blue to red)
    colormap = cm.get_cmap('turbo')  # Rainbow colormap: blue -> cyan -> green -> yellow -> red
    colors = colormap(normalized_distances)
    
    # Convert to 0-255 range for VTK
    colors_255 = (colors[:, :3] * 255).astype(np.uint8)
    
    return colors_255, mean_dist, std_dist


def visualize_with_vtk_interactive(vertices, faces, distances, vertex_mapping=None, title="Mesh Distance Heatmap"):
    """Interactive VTK visualization with range cycling and text display."""
    print("Creating interactive VTK visualization...")
    
    # Calculate distance statistics once
    mean_dist = np.mean(distances[distances > 0])
    std_dist = np.std(distances[distances > 0])
    
    # Define the range modes: [multiplier, description]
    range_modes = [
        (3.0, "3 STD"),
        (2.0, "2 STD"), 
        (1.0, "1 STD"),
        (10.0, "Fixed 10"),
        (5.0, "Fixed 5"),
        (3.0, "Fixed 3")
    ]
    current_mode_idx = 0
    
    def get_max_distance(mode_idx):
        """Get max distance value based on current mode."""
        multiplier, desc = range_modes[mode_idx]
        if desc == "Fixed 10":
            return 10.0
        elif desc == "Fixed 5":
            return 5.0
        elif desc == "Fixed 3":
            return 3.0
        else:
            return multiplier * std_dist
    
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
    
    # Create mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Create renderer and render window
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)
    
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(title)
    render_window.SetSize(1000, 700)
    
    # Create text actors
    range_text = vtk.vtkTextActor()
    range_text.SetPosition(10, 650)
    range_text.GetTextProperty().SetFontSize(16)
    range_text.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    renderer.AddActor2D(range_text)
    
    instruction_text = vtk.vtkTextActor()
    instruction_text.SetInput("Press TAB key to rotate ranges")
    instruction_text.SetPosition(10, 620)
    instruction_text.GetTextProperty().SetFontSize(12)
    instruction_text.GetTextProperty().SetColor(0.8, 0.8, 0.8)
    renderer.AddActor2D(instruction_text)
    
    # Create scalar bar (color gradient)
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetTitle("Distance")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetPosition(0.85, 0.1)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.8)
    scalar_bar.GetTitleTextProperty().SetColor(1.0, 1.0, 1.0)
    scalar_bar.GetLabelTextProperty().SetColor(1.0, 1.0, 1.0)
    renderer.AddActor2D(scalar_bar)
    
    def update_visualization():
        """Update colors and text based on current mode."""
        max_distance = get_max_distance(current_mode_idx)
        
        # Set distances as scalars
        full_distances = np.zeros(len(vertices))
        if vertex_mapping is not None:
            for i, orig_idx in enumerate(vertex_mapping):
                if i < len(distances):
                    full_distances[orig_idx] = distances[i]
        else:
            full_distances = distances
        
        vtk_distances = numpy_support.numpy_to_vtk(full_distances)
        vtk_distances.SetName("Distances")
        polydata.GetPointData().SetScalars(vtk_distances)
        
        # Configure mapper for scalar mapping
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarRange(0, max_distance)
        
        # Create lookup table
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        colormap = cm.get_cmap('turbo')
        for i in range(256):
            rgba = colormap(i / 255.0)
            lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], 1.0)
        lut.SetRange(0, max_distance)
        lut.Build()
        
        mapper.SetLookupTable(lut)
        scalar_bar.SetLookupTable(lut)
        
        # Update text
        mode_desc = range_modes[current_mode_idx][1]
        range_text.SetInput(f"Range: 0 to {max_distance:.3f} ({mode_desc})")
        
        # Refresh display
        render_window.Render()
    
    # Key press callback
    def key_press_callback(obj, event):
        nonlocal current_mode_idx
        key = obj.GetKeySym()
        if key == "Tab":
            current_mode_idx = (current_mode_idx + 1) % len(range_modes)
            update_visualization()
            print(f"Switched to range mode: {range_modes[current_mode_idx][1]}")
    
    # Set up interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.AddObserver("KeyPressEvent", key_press_callback)
    
    # Set up camera
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Zoom(1.2)
    
    # Initial visualization update
    update_visualization()
    
    # Start interaction
    render_window.Render()
    interactor.Start()


def main():
    """Main function to execute the mesh comparison workflow."""
    print("=== Mesh Distance Comparison Tool ===")
    
    # Load and preprocess both scans
    data1 = load_and_preprocess_nifti(SCAN1_PATH, THRESHOLD1, VOXEL_SIZE)
    data2 = load_and_preprocess_nifti(SCAN2_PATH, THRESHOLD2, VOXEL_SIZE)
    
    # Generate meshes using marching cubes
    vertices1, faces1 = generate_mesh_marching_cubes(data1, THRESHOLD1, VOXEL_SIZE)
    vertices2, faces2 = generate_mesh_marching_cubes(data2, THRESHOLD2, VOXEL_SIZE)
    
    # Extract outer shells
    outer_vertices1, outer_faces1, vertex_mapping1 = get_outer_mesh_with_mapping(vertices1, faces1)
    outer_vertices2, outer_faces2, vertex_mapping2 = get_outer_mesh_with_mapping(vertices2, faces2)

    if QUICK_SIMPLIFICATION and len(outer_vertices1) > 400000:
        # Calculate simplification factor based on number of vertices
        SIMPLIFICATION_FACTOR = max(min(1, 1-(400000/len(outer_vertices1))), 0)

        print(f"Simplifying mesh 1 with factor of {SIMPLIFICATION_FACTOR}...")
        outer_vertices1, outer_faces1 = fast_simplification.simplify(outer_vertices1, outer_faces1, SIMPLIFICATION_FACTOR)
        print(f"Simplified mesh 1: {len(outer_vertices1)} vertices, {len(outer_faces1)} faces")

    if QUICK_SIMPLIFICATION and len(outer_vertices2) > 400000:
        
        SIMPLIFICATION_FACTOR = max(min(1, 1-(400000/len(outer_vertices2))), 0)

        print(f"Simplifying mesh 2 with factor of {SIMPLIFICATION_FACTOR}...")
        outer_vertices2, outer_faces2 = fast_simplification.simplify(outer_vertices2, outer_faces2, SIMPLIFICATION_FACTOR)
        print(f"Simplified mesh 2: {len(outer_vertices2)} vertices, {len(outer_faces2)} faces")
    
    # Calculate distances from mesh 1 outer shell to mesh 2 outer shell
    distances = calculate_landmark_distances(outer_vertices1, outer_vertices2, outer_faces2)
    
    print(f"Distance statistics:")
    print(f"Mean distance: {np.mean(distances):.3f}")
    print(f"Max distance: {np.max(distances):.3f}")
    print(f"Standard deviation: {np.std(distances):.3f}")
    
    # Launch interactive visualization
    print("Launching interactive visualization...")
    print("Use TAB key to cycle through different distance ranges!")
    visualize_with_vtk_interactive(outer_vertices1, outer_faces1, distances, None)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main() 