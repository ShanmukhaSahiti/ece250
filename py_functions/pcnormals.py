import open3d as o3d
import numpy as np

def pcnormals_open3d(verts, search_radius=0.1, max_nn=30, k_neighbors_if_no_radius=12):
    """
    Estimates normals for a point cloud using Open3D.
    Args:
        verts (np.ndarray): Nx3 array of vertices.
        search_radius (float): Radius for KDTreeSearchParamHybrid.
        max_nn (int): Max neighbors for KDTreeSearchParamHybrid.
        k_neighbors_if_no_radius (int): Number of neighbors if radius search fails or is not preferred.
                                         Corresponds to Matlab's pcnormals(ptCloud, K)

    Returns:
        np.ndarray: Nx3 array of normals. Returns zeros if estimation fails.
    """
    if verts.shape[0] == 0:
        return np.empty((0,3), dtype=float)
    if verts.shape[1] != 3:
        raise ValueError("Vertices must be an Nx3 array.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)

    if not pcd.has_points():
        return np.zeros_like(verts)

    try:
        # Behavior closer to pcnormals(pointCloud(verts), K) might be estimate_normals with KDTreeSearchParamKNN
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors_if_no_radius))
        
        # The original Matlab was pcnormals(pointCloud(verts),12) which suggests K-nearest neighbors.
        # Let's default to KNN based on the Matlab function signature if K is provided.
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors_if_no_radius))
        
        # Optional: Orient normals consistently (e.g., towards viewpoint or origin)
        # pcd.orient_normals_consistent_tangent_plane(k=k_neighbors_if_no_radius) 
        # Or orient towards a camera location if available
        # pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))
        
        normals = np.asarray(pcd.normals)
        if normals.shape[0] != verts.shape[0]:
            print("Warning: Normal estimation returned unexpected number of normals. Falling back to zeros.")
            return np.zeros_like(verts)
        return normals
    except Exception as e:
        print(f"Error during Open3D normal estimation: {e}. Returning zero normals.")
        return np.zeros_like(verts) 