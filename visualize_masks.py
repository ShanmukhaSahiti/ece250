import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os

def load_mask(file_path):
    """
    Loads a mask from a .mat file.

    Args:
        file_path (str): The path to the .mat file.

    Returns:
        numpy.ndarray: The mask array, or None if loading fails or mask data is not found.
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        # Try to find the mask data. .mat files can store multiple variables.
        # We'll look for a variable named 'mask' or 'segmentation'.
        # If not found, we'll try to get the largest array in the .mat file,
        # as that's often the image or mask data.
        
        possible_keys = ['mask', 'segmentation', 'Mask', 'Segmentation']
        mask_array = None

        for key in possible_keys:
            if key in mat_data:
                mask_array = mat_data[key]
                break
        
        if mask_array is None:
            # If no common key is found, find the largest numerical array
            largest_array = None
            max_size = 0
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray) and value.ndim >= 2: # Check for at least 2D arrays
                    if value.size > max_size:
                        largest_array = value
                        max_size = value.size
            mask_array = largest_array

        if mask_array is not None:
            print(f"Successfully loaded mask from {file_path}. Mask shape: {mask_array.shape}")
            return mask_array
        else:
            print(f"Could not find mask data in {file_path}. Available keys: {list(mat_data.keys())}")
            return None
    except Exception as e:
        print(f"Error loading .mat file {file_path}: {e}")
        return None

def display_mask(mask_array, title="Mask Visualization"):
    """
    Displays a mask using matplotlib.

    Args:
        mask_array (numpy.ndarray): The mask array to display.
        title (str): The title for the plot.
    """
    if mask_array is None:
        print("Mask array is None, cannot display.")
        return

    plt.imshow(mask_array, cmap='gray') # 'gray' colormap is typical for masks
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # Ensure the script can find the video_meshes directory relative to its location
    # This assumes the script is in the workspace root and video_meshes is a subdirectory.
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    # Example usage:
    # Replace with the actual path to one of your mask files
    # example_mask_file = "video_meshes/lifting/v1_mat_mask/2502-2_70340_mask.mat"
    
    # Let's construct the path more robustly
    example_mask_relative_path = "video_meshes/lifting/v1_mat_mask/2502-2_70340_mask.mat"
    example_mask_file_path = os.path.join(base_dir, example_mask_relative_path)

    print(f"Attempting to load mask: {example_mask_file_path}")

    if not os.path.exists(example_mask_file_path):
        print(f"Error: Mask file not found at {example_mask_file_path}")
        print(f"Please check the path. Current working directory: {os.getcwd()}")
        print("Expected structure: <workspace_root>/visualize_masks.py and <workspace_root>/video_meshes/...")
    else:
        mask = load_mask(example_mask_file_path)
        if mask is not None:
            display_mask(mask, title=f"Mask: {os.path.basename(example_mask_file_path)}")
        else:
            print("Failed to load or display the mask.")

    # You can extend this to loop through all masks in a directory
    # mask_dir = os.path.join(base_dir, "video_meshes/lifting/v1_mat_mask/")
    # if os.path.isdir(mask_dir):
    #     for f_name in os.listdir(mask_dir):
    #         if f_name.endswith('_mask.mat'):
    #             mask_path = os.path.join(mask_dir, f_name)
    #             print(f"Processing {mask_path}...")
    #             current_mask = load_mask(mask_path)
    #             if current_mask is not None:
    #                 display_mask(current_mask, title=f"Mask: {f_name}")
    # else:
    #     print(f"Mask directory not found: {mask_dir}") 