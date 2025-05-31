import numpy as np
import scipy.io
import scipy.signal
from scipy.stats import find_peaks
from .get_local_coordinate_system import get_local_coordinate_system
from .get_original_pt_location import get_original_pt_location

def mesh_alignment_algorithm(all_verts_time, all_joints_time, person_box_time, person_mask_time, 
                           cams_time, im_shape, cls, cls_category, vid):
    """
    Implements the mesh alignment algorithm via eigen analysis as in sec 3.2 (or 4.2.2) of the paper.
    
    Args:
        all_verts_time (np.ndarray): Coordinates of all mesh points with time before alignment
        all_joints_time (np.ndarray): Coordinates of all joints with time before alignment
        person_box_time (np.ndarray): The bounding box of the person in the video frames
        person_mask_time (list): The binary mask of the person in the video frames
        cams_time (np.ndarray): Camera parameter in all frames
        im_shape (tuple): Dimensions of the video frames in pixels
        cls (str): The activity to be adjusted
        cls_category (int): The category of the activity
        vid (str): Which video of the activity
        
    Returns:
        tuple: (verts, joints, period_start_frames)
            - verts: Coordinates of all mesh points with time after alignment
            - joints: Coordinates of all joints with time after alignment
            - period_start_frames: The frames that indicate the start of each period
    """
    # Load the indices of different body parts in the meshes
    # These should be loaded from .mat files in the MeshInfoMatFiles directory
    try:
        torso_vts_idx = scipy.io.loadmat('MeshInfoMatFiles/torso_vts_idx.mat')['torso_vts_idx'].flatten() - 1
        left_toes_idx = scipy.io.loadmat('MeshInfoMatFiles/left_toes_idx.mat')['left_toes_idx'].flatten() - 1
        right_toes_idx = scipy.io.loadmat('MeshInfoMatFiles/right_toes_idx.mat')['right_toes_idx'].flatten() - 1
        right_fingers_idx = scipy.io.loadmat('MeshInfoMatFiles/right_fingers_idx.mat')['right_fingers_idx'].flatten() - 1
        left_fingers_idx = scipy.io.loadmat('MeshInfoMatFiles/left_fingers_idx.mat')['left_fingers_idx'].flatten() - 1
        leg1_idx = scipy.io.loadmat('MeshInfoMatFiles/leg1_idx.mat')['leg1_idx'].flatten() - 1
        leg2_idx = scipy.io.loadmat('MeshInfoMatFiles/leg2_idx.mat')['leg2_idx'].flatten() - 1
        left_foot_idx = scipy.io.loadmat('MeshInfoMatFiles/left_foot_idx.mat')['left_foot_idx'].flatten() - 1
        right_foot_idx = scipy.io.loadmat('MeshInfoMatFiles/right_foot_idx.mat')['right_foot_idx'].flatten() - 1
        left_thigh_idx = scipy.io.loadmat('MeshInfoMatFiles/left_thigh_idx.mat')['left_thigh_idx'].flatten() - 1
        right_thigh_idx = scipy.io.loadmat('MeshInfoMatFiles/right_thigh_idx.mat')['right_thigh_idx'].flatten() - 1
        hip_idx = scipy.io.loadmat('MeshInfoMatFiles/hip_idx.mat')['hip_idx'].flatten() - 1
    except Exception as e:
        print(f"Error loading body part indices: {e}")
        return all_verts_time, all_joints_time, np.array([])

    front_torso_idx = 3050  # 0-based index (3051-1)
    back_torso_idx = 460    # 0-based index (461-1)

    num_frames = all_verts_time.shape[2]  # total number of frames in video

    # Standing actions w/ one foot/both feet static
    if cls_category in [1, 2]:
        verts = all_verts_time.copy()
        joints = all_joints_time.copy()
        
        # Getting the reference frame (the frame in which the person is standing)
        person_box_height = person_box_time[2,:] - person_box_time[0,:]
        person_box_width = person_box_time[3,:] - person_box_time[1,:]
        
        if cls_category == 1:
            standing_frame = np.argmax(person_box_height)
            # Compute autocorrelation
            person_box_ACF = np.correlate(person_box_height, person_box_height, mode='full')
            person_box_ACF = person_box_ACF[len(person_box_height)-1:]
            person_box_ACF = person_box_ACF[:len(person_box_height)-50]
            peak_locs, _ = find_peaks(person_box_ACF, distance=25)
            action_period = np.mean(np.diff(np.concatenate([[0], peak_locs])))
        else:
            standing_frame = np.argmax(person_box_height - person_box_width)
            try:
                # Compute autocorrelation
                person_box_ACF = np.correlate(person_box_height - person_box_width, 
                                           person_box_height - person_box_width, mode='full')
                person_box_ACF = person_box_ACF[len(person_box_height)-1:]
                person_box_ACF = person_box_ACF[:len(person_box_height)-50]
            except:  # very short video
                person_box_ACF = np.correlate(person_box_height - person_box_width,
                                           person_box_height - person_box_width, mode='full')
                person_box_ACF = person_box_ACF[len(person_box_height)-1:]
                person_box_ACF = person_box_ACF[:len(person_box_height)-1]
            
            peak_locs, _ = find_peaks(person_box_ACF, distance=25)
            action_period = np.mean(np.diff(np.concatenate([[0], peak_locs])))
        
        if cls == 'lateral lunge':
            action_period = action_period * 2  # since we take right and left lunges to be one period
        
        # Get the mesh of the reference frame
        standing_frame_vts = verts[:,:,standing_frame]
        
        # Get the LCS for the reference frame (via eigen analysis)
        localX, localY, localZ = get_local_coordinate_system(
            standing_frame_vts,
            np.concatenate([torso_vts_idx, leg1_idx, leg2_idx]),
            front_torso_idx,
            back_torso_idx,
            left_toes_idx,
            show_plot=False
        )
        
        # Get ankle positions in original frame
        left_ankle_original_standing = get_original_pt_location(
            joints[0,:,standing_frame],
            person_box_time[:,standing_frame],
            cams_time[:,standing_frame],
            im_shape
        )
        right_ankle_original_standing = get_original_pt_location(
            joints[5,:,standing_frame],
            person_box_time[:,standing_frame],
            cams_time[:,standing_frame],
            im_shape
        )
        
        # Test by aligning the standing person's mesh and getting where the feet are in the GCS
        standing_frame_vts_adjusted = (standing_frame_vts - np.mean(standing_frame_vts, axis=0)) @ np.column_stack([localX, localY, localZ])
        left_ankle_fixed = np.mean(standing_frame_vts_adjusted[left_foot_idx,:], axis=0) - \
                          np.mean(standing_frame_vts_adjusted[np.concatenate([left_foot_idx, right_foot_idx]),:], axis=0)
        right_ankle_fixed = np.mean(standing_frame_vts_adjusted[right_foot_idx,:], axis=0) - \
                           np.mean(standing_frame_vts_adjusted[np.concatenate([left_foot_idx, right_foot_idx]),:], axis=0)
        
        for i in range(num_frames):
            # Rotate/align the frame
            verts[:,:,i] = (verts[:,:,i] - np.mean(verts[:,:,i], axis=0)) @ np.column_stack([localX, localY, localZ])
            
            if cls == 'lateral lunge':
                # For lateral lunges, make sure the person does not bend forward in the alignment process
                U, _, _ = np.linalg.svd(verts[:,:,i].T, full_matrices=False)
                which_col = np.argmax(np.abs(U[0,:]))
                U = U * np.sign(U[0,which_col])
                theta = max(np.degrees(np.arctan2(U[2,which_col], U[0,which_col])), 0)
                
                rotation_matrix = np.array([
                    [np.cos(np.radians(theta)), 0, np.sin(np.radians(theta))],
                    [0, 1, 0],
                    [-np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]
                ])
                verts[:,:,i] = verts[:,:,i] @ rotation_matrix.T
                
                left_foot_check = np.linalg.norm(
                    left_ankle_original_standing - 
                    get_original_pt_location(joints[0,:,i], person_box_time[:,i], cams_time[:,i], im_shape)
                )
                right_foot_check = np.linalg.norm(
                    right_ankle_original_standing - 
                    get_original_pt_location(joints[5,:,i], person_box_time[:,i], cams_time[:,i], im_shape)
                )
                
                if left_foot_check >= right_foot_check:
                    verts[:,:,i] = verts[:,:,i] - np.mean(verts[left_foot_idx,:,i], axis=0) + left_ankle_fixed
                else:
                    verts[:,:,i] = verts[:,:,i] - np.mean(verts[right_foot_idx,:,i], axis=0) + right_ankle_fixed
            
            if cls == 'stiff-leg deadlift':
                # For stiff-leg deadlift, make sure the legs stay up-right
                _, _, localZcurr = get_local_coordinate_system(
                    verts[:,:,i],
                    np.concatenate([leg1_idx, leg2_idx]),
                    front_torso_idx,
                    back_torso_idx,
                    left_toes_idx,
                    show_plot=False
                )
                theta = np.degrees(np.arccos(np.abs(localZcurr[2])/np.linalg.norm([localZcurr[0], localZcurr[2]]))) * 0.75
                
                rotation_matrix = np.array([
                    [np.cos(np.radians(theta)), 0, np.sin(np.radians(theta))],
                    [0, 1, 0],
                    [-np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]
                ])
                verts[:,:,i] = verts[:,:,i] @ rotation_matrix.T
                verts[:,:,i] = verts[:,:,i] - np.mean(verts[np.concatenate([left_foot_idx, right_foot_idx]),:,i], axis=0)
        
        # Calculate period start frames
        period_start_frames = np.arange(-action_period, num_frames + action_period + 1, action_period)
        frameshift = np.min(np.abs(standing_frame - period_start_frames))
        frameshift_loc = np.argmin(np.abs(standing_frame - period_start_frames))
        period_start_frames = period_start_frames + frameshift * np.sign(standing_frame - period_start_frames[frameshift_loc])
        period_start_frames = period_start_frames[(period_start_frames >= 0) & (period_start_frames < num_frames)]
    
    # Sit up
    elif cls_category == 8:
        verts = all_verts_time.copy()
        joints = all_joints_time.copy()
        
        # Get the reference frame as the widest frame (person lying down)
        person_box_width = person_box_time[3,:] - person_box_time[1,:]
        lying_frame = np.argmax(person_box_width)
        
        lying_frame_vts = verts[:,:,lying_frame]
        
        # Get the LCS
        localX, localY, localZ = get_local_coordinate_system(
            lying_frame_vts,
            np.concatenate([torso_vts_idx, right_foot_idx, left_foot_idx]),
            front_torso_idx,
            back_torso_idx,
            left_toes_idx,
            show_plot=False
        )
        
        # Transform all frames (note now the local X axis points to global Z)
        for i in range(num_frames):
            verts[:,:,i] = (verts[:,:,i] - np.mean(verts[:,:,i], axis=0)) @ np.column_stack([-localZ, localY, localX])
            verts[:,:,i] = verts[:,:,i] - np.mean(verts[hip_idx,:,i], axis=0)
        
        # Compute autocorrelation and find peaks
        person_box_ACF = np.correlate(person_box_width, person_box_width, mode='full')
        person_box_ACF = person_box_ACF[len(person_box_width)-1:]
        person_box_ACF = person_box_ACF[:len(person_box_width)-50]
        peak_locs, _ = find_peaks(person_box_ACF, distance=25)
        action_period = np.mean(np.diff(np.concatenate([[0], peak_locs])))
        
        # Calculate period start frames
        period_start_frames = np.arange(-action_period, num_frames + action_period + 1, action_period)
        frameshift = np.min(np.abs(lying_frame - period_start_frames))
        frameshift_loc = np.argmin(np.abs(lying_frame - period_start_frames))
        period_start_frames = period_start_frames + frameshift * np.sign(lying_frame - period_start_frames[frameshift_loc])
        period_start_frames = period_start_frames[(period_start_frames >= 0) & (period_start_frames < num_frames)]
    
    else:  # unidentified category
        verts = all_verts_time.copy()
        joints = all_joints_time.copy()
        period_start_frames = np.array([])
    
    return verts, joints, period_start_frames 