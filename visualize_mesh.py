import scipy.io
import numpy as np
import open3d as o3d
import sys
import os

def visualize_hmr_mesh(mat_file_path):
    """
    Loads and visualizes a human mesh from an HMR output .mat file.
    The .mat file is expected to contain 'verts'.
    If 'faces' is present, it will draw a mesh, otherwise a point cloud.
    """
    try:
        data = scipy.io.loadmat(mat_file_path)
        print(f"Successfully loaded .mat file: {mat_file_path}")
        print(f"Keys in .mat file: {list(data.keys())}")
    except Exception as e:
        print(f"Error loading .mat file '{mat_file_path}': {e}")
        return

    if 'verts' not in data:
        print(f"Error: 'verts' key not found in '{mat_file_path}'")
        return
    
    vertices = data['verts']
    
    # Squeeze the array to remove single-dimensional entries from the shape
    if isinstance(vertices, np.ndarray):
        original_shape = vertices.shape
        vertices = np.squeeze(vertices)
        print(f"Shape of 'verts' (original: {original_shape}, after squeeze: {vertices.shape})")

    if not isinstance(vertices, np.ndarray) or vertices.ndim != 2 or vertices.shape[1] != 3:
        print(f"Error: 'verts' data is not in the expected (N, 3) NumPy array format after squeeze. Current shape: {vertices.shape}")
        return

    print(f"Loaded {vertices.shape[0]} vertices (from 'verts' key, after squeeze).")

    geometry_to_draw = None
    window_title = f"Visualization: {os.path.basename(mat_file_path)}"

    if 'faces' in data:
        faces = data['faces']
        if faces.min() == 1:
            print("Faces appear to be 1-indexed, converting to 0-indexed.")
            faces = faces - 1
        
        if isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3:
            print(f"Loaded {faces.shape[0]} faces.")
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
            geometry_to_draw = mesh
            window_title = f"Mesh: {os.path.basename(mat_file_path)}"
            print("Preparing to visualize mesh...")
        else:
            print(f"Warning: 'faces' data is not valid. Visualizing as point cloud.")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            pcd.paint_uniform_color([0.5, 0.5, 0.9])
            geometry_to_draw = pcd
            window_title = f"Point Cloud (faces invalid): {os.path.basename(mat_file_path)}"
            print("Preparing to visualize point cloud (faces invalid)...")
    else:
        print("No 'faces' key found. Visualizing as point cloud.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.paint_uniform_color([0.5, 0.9, 0.5]) 
        geometry_to_draw = pcd
        window_title = f"Point Cloud (no faces key): {os.path.basename(mat_file_path)}"
        print("Preparing to visualize point cloud (no faces key)...")

    if geometry_to_draw:
        try:
            o3d.visualization.draw_geometries([geometry_to_draw], window_name=window_title)
            print("Visualization window closed.")
        except Exception as e:
            print(f"Error during Open3D visualization: {e}")
            print("This might happen if you are running in a headless environment or X11 forwarding is not set up.")
            print("Make sure you are running this script from a terminal within your Ubuntu VM's desktop environment.")
    else:
        print("No geometry could be prepared for visualization.")

if __name__ == "__main__":
    default_mat_file = None 

    if len(sys.argv) < 2:
        if default_mat_file and os.path.exists(default_mat_file):
            print(f"No file path provided. Visualizing default file: {default_mat_file}")
            visualize_hmr_mesh(default_mat_file)
        else:
            print("Usage: python visualize_mesh.py <path_to_your_mesh.mat_file>")
            if default_mat_file:
                 print(f"(Default file not found: {default_mat_file})")
    else:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'")
        else:
            visualize_hmr_mesh(file_path)