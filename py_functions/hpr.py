import numpy as np
import scipy.spatial

def hpr_equivalent(p, C, param):
    """
    Approximate Python equivalent of HPR function.
    Based on the description and Matlab code.
    Args:
        p (np.ndarray): NxD point cloud.
        C (np.ndarray): 1xD viewpoint.
        param (float): Parameter for the algorithm.
    Returns:
        np.ndarray: Indices of p that are visible from C.
    """
    num_pts, dim = p.shape
    if num_pts == 0:
        return np.array([], dtype=int)
        
    p_centered = p - C  # Move C to the origin. C should be 1xD, p is NxD.
    norm_p = np.linalg.norm(p_centered, axis=1) # Calculate ||p||, result is (N,)
    
    # Sphere radius
    max_norm_p = np.max(norm_p)
    if max_norm_p == 0: # All points are at the viewpoint C
        return np.arange(num_pts)
        
    R_val = max_norm_p * (10**param)
    R_col_vec = np.full((num_pts, 1), R_val) # R as a column vector (N,1)
    
    # Spherical flipping
    norm_p_col = norm_p.reshape(-1, 1) # norm_p as a column vector (N,1)
    
    epsilon = 1e-9 # Avoid division by zero
    denominator = norm_p_col + epsilon
    
    # P = p + 2*(R-normp) .* p ./ normp (Matlab element-wise)
    # p_centered is points relative to origin C
    # R-normp_col is (N,1), p_centered is (N,D), norm_p_col is (N,1)
    # scalar_factor = 2 * (R_col_vec - norm_p_col) / denominator # This is (N,1)
    # P_flipped = p_centered + scalar_factor * p_centered # This also works
    P_flipped = p_centered + 2 * (R_col_vec - norm_p_col) * p_centered / denominator

    # Convex hull
    # Append origin (0,0,...,0) which was C before centering
    points_for_hull = np.vstack([P_flipped, np.zeros((1, dim))])
    
    try:
        # Qhull can fail for degenerate cases (e.g. colinear points in 2D, coplanar in 3D)
        if dim < 2 or num_pts < dim +1: # Not enough points/dimensions for convhull
             # print(f"HPR: Not enough points or dimensions for convex hull. Dims: {dim}, Points: {num_pts}. Returning all points.")
             return np.arange(num_pts)
        hull = scipy.spatial.ConvexHull(points_for_hull)
        visible_pt_indices = np.unique(hull.vertices)
        # Remove the index of the added origin point (num_pts is its 0-based index after vstack)
        visible_pt_indices = visible_pt_indices[visible_pt_indices != num_pts]
    except scipy.spatial.qhull.QhullError as e:
        print(f"QhullError in HPR: {e}. Num_pts={num_pts}, Dim={dim}. Returning all points as fallback.")
        # Fallback: consider all points visible or handle more gracefully
        visible_pt_indices = np.arange(num_pts)
    except Exception as e:
        print(f"Unexpected error in HPR convex hull: {e}. Returning all points.")
        visible_pt_indices = np.arange(num_pts)
        
    return visible_pt_indices.astype(int) 