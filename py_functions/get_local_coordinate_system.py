import numpy as np

def get_local_coordinate_system(verts, torso_vts_idx, front_torso_idx, back_torso_idx, left_toes_idx, show_plot=False, ax=None):
    """
    Calculates the local coordinate system for a set of vertices.
    Translated from get_local_coordinate_system.m

    Args:
        verts (np.ndarray): Nx3 array of vertices.
        torso_vts_idx (np.ndarray): Indices for torso vertices (0-indexed).
        front_torso_idx (int): Index for a front torso vertex (0-indexed).
        back_torso_idx (int): Index for a back torso vertex (0-indexed).
        left_toes_idx (np.ndarray): Indices for left toes vertices (0-indexed).
                                     Only the first element is used if provided as an array.
        show_plot (bool): If True, attempts to plot (requires ax to be provided).
        ax (matplotlib.axes.Axes, optional): Matplotlib 3D axes for plotting if show_plot is True.

    Returns:
        tuple: (localX, localY, localZ) each as a 1D np.ndarray of shape (3,).
    """
    # Ensure indices are integer for array indexing
    torso_vts_idx = np.asarray(torso_vts_idx, dtype=int).flatten()
    front_torso_idx = int(front_torso_idx)
    back_torso_idx = int(back_torso_idx)
    # Ensure left_toes_idx is an array and we take the first element's index
    left_toes_first_idx = int(np.asarray(left_toes_idx, dtype=int).flatten()[0])

    torso_vts = verts[torso_vts_idx, :]
    if torso_vts.shape[0] == 0:
        # Fallback if no torso vertices are indexed, return standard axes
        print("Warning: No torso vertices found for LCS. Returning standard axes.")
        return np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
        
    torso_centroid = np.mean(torso_vts, axis=0)
    
    # Eigen decomposition of the covariance matrix of the centered torso vertices
    # (torso_vts-torso_centroid)' * (torso_vts-torso_centroid)
    centered_torso_vts = torso_vts - torso_centroid
    covariance_matrix = centered_torso_vts.T @ centered_torso_vts # or np.cov(centered_torso_vts.T)
    
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    # Matlab's sort(diag(DD),'descend') gives sorted values and original indices.
    # np.argsort sorts in ascending, so we reverse it [::-1].
    sorted_indices = np.argsort(eigen_values)[::-1] # Indices for descending sort
    
    # localX: eigenvector for the smallest eigenvalue (axis_strength(3) in Matlab 1-indexed)
    # Corresponds to sorted_indices[2] in Python 0-indexed
    localX = eigen_vectors[:, sorted_indices[2]]
    
    # Orient localX: (verts(front_torso_idx,:) - verts(back_torso_idx,:))*localX < 0
    # This checks if the vector from back_torso_idx to front_torso_idx is in the opposite direction of localX.
    # If so, flip localX.
    orientation_vector_X = verts[front_torso_idx, :] - verts[back_torso_idx, :]
    if np.dot(orientation_vector_X, localX) < 0:
        localX = -localX
    
    # localZ: eigenvector for the largest eigenvalue (axis_strength(1) in Matlab)
    # Corresponds to sorted_indices[0] in Python
    localZ = eigen_vectors[:, sorted_indices[0]]
    
    # Orient localZ: (verts(front_torso_idx,:) - verts(left_toes_idx(1),:))*localZ < 0
    # Vector from a left toe point to the front torso point.
    orientation_vector_Z = verts[front_torso_idx, :] - verts[left_toes_first_idx, :]
    if np.dot(orientation_vector_Z, localZ) < 0:
        localZ = -localZ
        
    localY = np.cross(localZ, localX)
    # Normalize to be sure, though eigenvectors from np.linalg.eig are usually normalized.
    localX = localX / np.linalg.norm(localX)
    localY = localY / np.linalg.norm(localY)
    localZ = localZ / np.linalg.norm(localZ)

    if show_plot and ax is not None:
        ax.scatter(verts[:,0], verts[:,1], verts[:,2], s=1, label='Vertices') # s=1 for '.' marker
        # Quiver for local axes
        ax.quiver(torso_centroid[0], torso_centroid[1], torso_centroid[2], 
                  localX[0], localX[1], localX[2], 
                  length=0.5, color='r', label='localX')
        ax.quiver(torso_centroid[0], torso_centroid[1], torso_centroid[2], 
                  localY[0], localY[1], localY[2], 
                  length=0.5, color='g', label='localY')
        ax.quiver(torso_centroid[0], torso_centroid[1], torso_centroid[2], 
                  localZ[0], localZ[1], localZ[2], 
                  length=0.5, color='k', label='localZ') # black for Z
        ax.set_xlabel('Global X')
        ax.set_ylabel('Global Y')
        ax.set_zlabel('Global Z')
        ax.legend()

    return localX, localY, localZ 