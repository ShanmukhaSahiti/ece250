#!/usr/bin/env python3
"""
This code implements the video-to-wifi simulation pipeline of the paper,
"Teaching RF to sense without RF training measurements",
published in IMWUT20 (vol. 4, issue 4), with application to the three
sample GYM activities (lateral lunge, sit up, and deadlift).

Copyright (c): H. Cai, B. Korany, and C. Karanam (UCSB, 2020)
"""

import os
import numpy as np
import scipy.io
import scipy.signal
import scipy.interpolate
import scipy.ndimage # For gaussian_filter and uniform_filter1d
from scipy.spatial import ConvexHull
import cv2 # OpenCV for imread, imgaussfilt (alternative)
from matplotlib import pyplot as plt
from natsort import natsorted # For natsortfiles equivalent
# from statsmodels.tsa.stattools import acf # For autocorr - to be added for mesh_alignment
import open3d as o3d # For point cloud processing if needed

# Import functions from py_functions module
from py_functions.get_action_name import get_action_name
from py_functions.remove_low_freq import remove_low_freq
from py_functions.hpr import hpr_equivalent
from py_functions.pcnormals import pcnormals_open3d
from py_functions.mesh_alignment_algorithm import mesh_alignment_algorithm

# Placeholder for functions that will be in py_functions and still need to be created/ported
# from py_functions.mesh_alignment_algorithm import mesh_alignment_algorithm
# from py_functions.get_local_coordinate_system import get_local_coordinate_system # Used by mesh_alignment
# from py_functions.get_original_pt_location import get_original_pt_location # Used by mesh_alignment


def pcnormals_equivalent(verts, k_neighbors=12):
    """
    Placeholder for Matlab's pcnormals(pointCloud(verts),12).
    This would typically use a library like Open3D or PyVista,
    or be implemented manually if the algorithm is simple.
    
    Args:
        verts (np.ndarray): Nx3 array of vertices.
        k_neighbors (int): Number of neighbors for normal estimation.

    Returns:
        np.ndarray: Nx3 array of normals.
    """
    print(f"Warning: pcnormals_equivalent is a placeholder. Needs implementation for {verts.shape} points.")
    # Simple placeholder: return zeros or random normals
    # For a real implementation with Open3D:
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(verts)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k_neighbors))
    # normals = np.asarray(pcd.normals)
    # return normals
    return np.random.rand(*verts.shape) * 2 - 1 # Random normals for now


# --- Start of functions to be moved to py_functions ---

def get_action_name(action_id):
    if action_id == 5:
        cls_name = 'lateral lunge'
        cls_category = 2
        available_vid = ['2', '5']
    elif action_id == 9:
        cls_name = 'sit up'
        cls_category = 8
        available_vid = ['3-4', '6']
    elif action_id == 10:
        cls_name = 'stiff-leg deadlift'
        cls_category = 1
        available_vid = ['4', '6']
    else:
        raise ValueError('Activity not found. Please choose one of the sample provided activities (5, 9, or 10).')
    return cls_name, cls_category, available_vid

def remove_low_freq(x, w):
    w = int(w) # ensure w is an integer for filter size
    filt = np.ones(w) / w
    if x.ndim == 1 or x.shape[0] == 1: # 1D array or row vector
        x_flat = x.flatten()
        lf = np.convolve(x_flat, filt, mode='same')
        if x.shape[0] > 1 and x.shape[1] == 1: # column vector
             lf = lf.reshape(x.shape)
    elif x.ndim == 2 : # 2D array
        # Applying 1D convolution along each column (Matlab's conv2(filt,1, x, 'same') behavior)
        lf = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=0, arr=x)
    else:
        raise ValueError("Input x must be 1D or 2D for remove_low_freq")
    
    y = x - lf
    return y

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
    p_centered = p - C  # Move C to the origin
    norm_p = np.linalg.norm(p_centered, axis=1) # Calculate ||p||
    
    # Sphere radius
    # Ensure R is a column vector for broadcasting if p_centered is NxD
    R_val = np.max(norm_p) * (10**param)
    R = np.full((num_pts, 1), R_val)
    
    # Spherical flipping
    # Ensure norm_p is also a column vector for element-wise operations
    norm_p_col = norm_p.reshape(-1, 1)
    
    # Avoid division by zero if any norm_p_col is zero
    # Add a small epsilon to norm_p_col in the denominator
    epsilon = 1e-9
    P_flipped = p_centered + 2 * (R - norm_p_col) * p_centered / (norm_p_col + epsilon)
    
    # Convex hull
    # Append origin (0,0,0) which was C before centering
    points_for_hull = np.vstack([P_flipped, np.zeros((1, dim))])
    
    try:
        hull = ConvexHull(points_for_hull)
        visible_pt_indices = np.unique(hull.vertices)
        # Remove the index of the added origin point (num_pts is its 0-based index)
        visible_pt_indices = visible_pt_indices[visible_pt_indices != num_pts]
    except scipy.spatial.qhull.QhullError as e:
        print(f"QhullError in HPR: {e}. This might happen with co-linear/co-planar points. Returning all points.")
        # Fallback or more sophisticated error handling might be needed
        visible_pt_indices = np.arange(num_pts)
        
    return visible_pt_indices


# --- End of functions to be moved to py_functions ---


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_info_path = os.path.join(script_dir, 'MeshInfoMatFiles')
    
    try:
        left_arm_idx_data = scipy.io.loadmat(os.path.join(mesh_info_path, 'left_arm_idx.mat'))
        left_arm_idx = left_arm_idx_data['left_arm_idx'].flatten() - 1
        
        right_arm_idx_data = scipy.io.loadmat(os.path.join(mesh_info_path, 'right_arm_idx.mat'))
        right_arm_idx = right_arm_idx_data['right_arm_idx'].flatten() - 1
        
        all_indices = np.arange(6890)
        arm_indices = np.concatenate([left_arm_idx, right_arm_idx])
        set_noarm = np.setdiff1d(all_indices, arm_indices)

    except FileNotFoundError as e:
        print(f"Error loading arm index .mat files: {e}. Please ensure 'MeshInfoMatFiles' directory is present.")
        return
    except KeyError as e:
        print(f"KeyError loading arm index .mat files: {e}. Check variable names in .mat files.")
        return

    fc = 5.18e9
    lambda_wave = 3e8 / fc
    Tx_pos_all = np.array([[2, 2, 0.76], [2, 2, 0.76], [-0.20, 0.25, 2.75]])
    Rx_pos_all = np.array([[2, -2, 0.76], [-2, 2, 0.76], [-0.20, -0.25, 2.75]])
    num_link = Tx_pos_all.shape[0]
    name_link = ['x', 'y', 'z']
    beamwidth_body_default = 40.0

    show_meshes = True
    activity_id = 5 

    if activity_id not in [5, 9, 10]:
        print('Activity not found. Please choose one of the sample provided activities (5, 9, or 10).')
        return

    # Load body part indices once, as they are static
    try:
        noarm_head_idx = scipy.io.loadmat(os.path.join(mesh_info_path, 'noarm_head_idx.mat'))['noarm_head_idx'].flatten() - 1
        noarm_torso_idx = scipy.io.loadmat(os.path.join(mesh_info_path, 'noarm_torso_idx.mat'))['noarm_torso_idx'].flatten() - 1
        noarm_left_leg_idx = scipy.io.loadmat(os.path.join(mesh_info_path, 'noarm_left_leg_idx.mat'))['noarm_left_leg_idx'].flatten() - 1
        noarm_right_leg_idx = scipy.io.loadmat(os.path.join(mesh_info_path, 'noarm_right_leg_idx.mat'))['noarm_right_leg_idx'].flatten() - 1
        all_part_indices_map = [noarm_head_idx, noarm_torso_idx, noarm_left_leg_idx, noarm_right_leg_idx]
    except FileNotFoundError as e:
        print(f"Error loading body part index .mat files: {e}. WiFi simulation cannot proceed.")
        return
    except KeyError as e:
        print(f"KeyError in body part index .mat files: {e}. WiFi simulation cannot proceed.")
        return

    _, _, available_activity_videos = get_action_name(activity_id)
    for vid_id_idx, vid_id_val_str in enumerate(available_activity_videos):
        
        print(f"Processing video index: {vid_id_idx}, video_id_string: {vid_id_val_str}")

        cls_name, cls_category, _ = get_action_name(activity_id)
        current_video_name_part = vid_id_val_str
        print(f'Class: {cls_name}\nVideo part: {current_video_name_part}')

        vid_prefix = f'v-{current_video_name_part}'
        video_frames_base = os.path.join(script_dir, 'video_frames')
        video_meshes_base = os.path.join(script_dir, 'video_meshes')
        folder_frame = os.path.join(video_frames_base, cls_name, vid_prefix)
        
        if not os.path.isdir(folder_frame):
            print(f"Frame folder not found: {folder_frame}")
            continue
            
        frame_all_files = natsorted([f for f in os.listdir(folder_frame) if os.path.isfile(os.path.join(folder_frame, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not frame_all_files:
            print(f"No image frames found in {folder_frame}")
            continue

        try:
            frame_one_path = os.path.join(folder_frame, frame_all_files[0])
            frame_one = cv2.imread(frame_one_path)
            if frame_one is None: raise IOError(f"Could not read image: {frame_one_path}")
            h_frame, w_frame, _ = frame_one.shape
        except Exception as e:
            print(f"Error loading first frame: {e}")
            continue
            
        im_shape = (h_frame, w_frame)
        folder_mesh = os.path.join(video_meshes_base, cls_name, f'{vid_prefix}_mat_mesh')
        folder_box = os.path.join(video_meshes_base, cls_name, f'{vid_prefix}_mat_mask')
        folder_cropped_im = os.path.join(video_meshes_base, cls_name, f'{vid_prefix}_cropped_im')

        if not os.path.isdir(folder_mesh):
            print(f"Mesh folder not found: {folder_mesh}")
            continue
        mesh_all_files = natsorted([f for f in os.listdir(folder_mesh) if f.endswith('.mat')])
        num_mesh = len(mesh_all_files)
        if num_mesh == 0:
            print(f"No mesh files (.mat) found in {folder_mesh}")
            continue

        cropped_im_all_files = []
        if os.path.isdir(folder_cropped_im):
            cropped_im_all_files = natsorted([f for f in os.listdir(folder_cropped_im) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        elif show_meshes:
            print(f"Cropped images folder {folder_cropped_im} not found. Visualization will be affected.")

        all_verts_time = np.zeros((6890, 3, num_mesh))
        all_joints_time = np.zeros((19, 3, num_mesh))
        person_box_time = np.full((4, num_mesh), np.nan)
        person_mask_time = [None] * num_mesh
        box_h_time = np.full(num_mesh, np.nan)
        box_c_time = np.full(num_mesh, np.nan)
        cams_time = np.zeros((3, num_mesh))
        cropped_images = [None] * num_mesh
        t_times = np.zeros(num_mesh)

        print('Obtaining mesh and (if applicable) mask data...')
        for iter_mesh in range(num_mesh):
            try:
                mesh_file_path = os.path.join(folder_mesh, mesh_all_files[iter_mesh])
                mat_data = scipy.io.loadmat(mesh_file_path)
                verts = np.squeeze(mat_data['verts'])
                joints3d = np.squeeze(mat_data['joints3d'])
                cams = mat_data['cams'].flatten()
                all_verts_time[:, :, iter_mesh] = verts
                all_joints_time[:, :, iter_mesh] = joints3d
                cams_time[:, iter_mesh] = cams
                
                if show_meshes and iter_mesh < len(cropped_im_all_files):
                    img_path = os.path.join(folder_cropped_im, cropped_im_all_files[iter_mesh])
                    cropped_images[iter_mesh] = cv2.imread(img_path)

                name_parts = mesh_all_files[iter_mesh].split('_')
                time_str_part = name_parts[2].split('.')[0]
                t_times[iter_mesh] = int(time_str_part)

                mask_file_name = f"{name_parts[0]}_{name_parts[1]}_{time_str_part}_mask.mat"
                mask_file_path = os.path.join(folder_box, mask_file_name)
                if os.path.exists(mask_file_path):
                    mask_data = scipy.io.loadmat(mask_file_path)
                    person_box = mask_data['person_box'].astype(float).flatten()
                    person_box_time[:, iter_mesh] = person_box
                    box_h_time[iter_mesh] = (h_frame - person_box[2]) / 500.0
                    box_c_time[iter_mesh] = ((person_box[3] + person_box[1]) / 2.0 - w_frame / 2.0) / 400.0
                    person_mask_time[iter_mesh] = mask_data['person_mask']
                # else: Mask data remains NaN if not found
            except Exception as e:
                print(f"Error processing frame {iter_mesh} ({mesh_all_files[iter_mesh]}): {e}")
                # Potentially skip this frame or handle error more gracefully
                # For now, allow loop to continue, NaNs might propagate

        if t_times.size > 0: t_times = t_times - t_times[0]

        all_verts_time = all_verts_time[:, [0, 2, 1], :] # x, z, y
        all_verts_time[:, :, 2] *= -1 # x, z, -y
        all_joints_time = all_joints_time[:, [0, 2, 1], :]
        all_joints_time[:, :, 2] *= -1
        all_verts_time_unaligned = all_verts_time.copy()
        
        print("Calling placeholder mesh_alignment_algorithm (currently returns unaligned data)")
        # TODO: Implement and call actual mesh_alignment_algorithm
        # verts_aligned, joints_aligned, period_start_frames = mesh_alignment_algorithm(
        #     all_verts_time, all_joints_time, person_box_time,
        #     person_mask_time, cams_time, im_shape,
        #     cls_name, cls_category, current_video_name_part 
        # )
        verts_aligned = all_verts_time 
        joints_aligned = all_joints_time 
        if num_mesh > 10: period_start_frames = np.array([0, int(num_mesh/2), num_mesh-1], dtype=int)
        elif num_mesh > 0: period_start_frames = np.array([0], dtype=int)
        else: period_start_frames = np.array([], dtype=int)
        if period_start_frames is None or period_start_frames.size == 0 or (np.isscalar(period_start_frames) and np.isnan(period_start_frames)) or np.all(np.isnan(period_start_frames)):
             period_start_frames = np.array([0]) if num_mesh > 0 else np.array([])

        ts_interp = 0.005
        tq_interp = np.array([])
        final_all_verts_time = verts_aligned # Default if no interpolation

        if t_times.size > 1 and t_times[-1] > t_times[0]:
            tq_interp = np.arange(t_times[0], t_times[-1] + ts_interp, ts_interp)
            if tq_interp.size == 0: tq_interp = np.array([t_times[0]]) if t_times.size > 0 else np.array([])
            
            vv, _, tt_orig = verts_aligned.shape
            all_verts_time_reshaped = verts_aligned.reshape(vv * 3, tt_orig).T
            unique_t_times, unique_indices = np.unique(t_times, return_index=True)

            if len(unique_t_times) >= 2 and tq_interp.size > 0:
                interp_func = scipy.interpolate.interp1d(unique_t_times, all_verts_time_reshaped[unique_indices,:], axis=0, kind='cubic', fill_value="extrapolate")
                all_verts_time_interp = interp_func(tq_interp).T 
                all_verts_time_resampled = all_verts_time_interp.reshape(vv, 3, len(tq_interp))
                window_len_smooth = 80
                if len(tq_interp) >= window_len_smooth:
                    final_all_verts_time = scipy.ndimage.uniform_filter1d(all_verts_time_resampled, size=window_len_smooth, axis=2, mode='nearest')
                else:
                    print(f"Data length ({len(tq_interp)}) < smoothing window ({window_len_smooth}). Using unsmoothed.")
                    final_all_verts_time = all_verts_time_resampled
            else:
                print("Not enough unique time points or tq_interp is empty. Skipping interpolation.")
                tq_interp = t_times # Use original time ticks
                final_all_verts_time = verts_aligned
        else:
            print("Time vector 't_times' not suitable for interpolation. Using original aligned verts.")
            tq_interp = t_times
            final_all_verts_time = verts_aligned

        period_start_times_vals = np.array([])
        if t_times.size > 0 and period_start_frames.size > 0 and num_mesh > 0:
            valid_period_indices = period_start_frames[(period_start_frames >= 0) & (period_start_frames < num_mesh)].astype(int)
            if valid_period_indices.size > 0 and np.arange(num_mesh).size == t_times.size: # Ensure consistent lengths for interp1d
                interp_time_for_periods = scipy.interpolate.interp1d(np.arange(num_mesh), t_times, fill_value="extrapolate", bounds_error=False)
                period_start_times_vals = interp_time_for_periods(valid_period_indices)

        if show_meshes and num_mesh > 0 and final_all_verts_time.shape[2] > 0:
            print('Showing the meshes...')
            fig_mesh_viz = plt.figure(figsize=(15, 4))
            for i_viz in range(min(num_mesh, final_all_verts_time.shape[2])):
                fig_mesh_viz.clf()
                ax1 = fig_mesh_viz.add_subplot(131)
                if i_viz < len(cropped_images) and cropped_images[i_viz] is not None:
                    ax1.imshow(cv2.cvtColor(cropped_images[i_viz], cv2.COLOR_BGR2RGB))
                ax1.set_title('Original Video'); ax1.axis('off')
                ax2 = fig_mesh_viz.add_subplot(132, projection='3d')
                verts_unaligned_frame_i = all_verts_time_unaligned[:, :, i_viz]
                ax2.scatter(verts_unaligned_frame_i[:, 0], verts_unaligned_frame_i[:, 1], verts_unaligned_frame_i[:, 2], s=1)
                ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
                ax2.set_xlim([-3, 3]); ax2.set_ylim([-3, 3]); ax2.set_zlim([-.5, 2.5])
                ax2.set_title('Unaligned Mesh'); ax2.view_init(elev=20, azim=45)
                ax3 = fig_mesh_viz.add_subplot(133, projection='3d')
                if tq_interp.size > 0 and t_times.size > i_viz:
                    j_viz = np.argmin(np.abs(tq_interp - t_times[i_viz]))
                    if j_viz < final_all_verts_time.shape[2]:
                         verts_aligned_frame_j = final_all_verts_time[:, :, j_viz]
                         ax3.scatter(verts_aligned_frame_j[:, 0], verts_aligned_frame_j[:, 1], verts_aligned_frame_j[:, 2], s=1)
                ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z')
                ax3.set_xlim([-3, 3]); ax3.set_ylim([-3, 3]); ax3.set_zlim([-.5, 2.5])
                ax3.set_title('Aligned Mesh'); ax3.view_init(elev=20, azim=45)
                plt.tight_layout(); plt.pause(0.05)
            plt.close(fig_mesh_viz)

        print('Simulating the WiFi signal...')
        if activity_id in [9, 10]: scaling_body = np.array([1, 1, 0, 0])
        elif activity_id == 5: scaling_body = np.array([1, 1, 1, 1])
        else: scaling_body = np.array([1, 1, 1, 1])

        sp_all_py = [np.array([]) for _ in range(num_link)]
        RS_all_py = [np.array([]) for _ in range(num_link)]
        T_s_py_glob, F_s_py_glob = np.array([]), np.array([])

        do_parts_interpolation_flag = True
        do_interp_parts_matrix = np.array([[0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,0]]) 
        interp_resolution_parts = np.array([0.01,0.02,0.01,0.01,0.01])
        active_links_for_sim = [0, 1, 2]
        epsilon_dist = 1e-9

        for iter_link_idx in active_links_for_sim:
            Tx_pos = Tx_pos_all[iter_link_idx, :]
            Rx_pos = Rx_pos_all[iter_link_idx, :]
            dist_Tx_Rx = np.linalg.norm(Tx_pos - Rx_pos)
            print(f'Generating WiFi Signal: Link {name_link[iter_link_idx]}...')
            
            num_time_samples_wifi = final_all_verts_time.shape[2]
            if num_time_samples_wifi == 0: continue
            RS_signal = np.zeros(num_time_samples_wifi, dtype=complex)

            for i_time_step in range(num_time_samples_wifi):
                verts_curr_all_parts_at_t = final_all_verts_time[:, :, i_time_step]
                verts_for_summation_curr_t = np.empty((0,3))
                scale_body_surface_for_summation_curr_t = np.array([])
                beamwidth_for_summation_curr_t = np.array([])

                if do_parts_interpolation_flag:
                    for iter_part_idx in range(len(scaling_body)):
                        if scaling_body[iter_part_idx] == 0: continue
                        current_global_part_indices = all_part_indices_map[iter_part_idx]
                        verts_part_potential = verts_curr_all_parts_at_t[current_global_part_indices, :]
                        if verts_part_potential.shape[0] == 0: continue

                        idx_visible_in_part_tx = hpr_equivalent(verts_part_potential, Tx_pos.reshape(1,-1), 2.25)
                        idx_visible_in_part_rx = hpr_equivalent(verts_part_potential, Rx_pos.reshape(1,-1), 2.25)
                        idx_visible_final_part = np.intersect1d(idx_visible_in_part_tx, idx_visible_in_part_rx)
                        verts_part_visible = verts_part_potential[idx_visible_final_part, :]
                        if verts_part_visible.shape[0] == 0: continue
                        
                        if iter_link_idx < do_interp_parts_matrix.shape[0] and \
                           iter_part_idx < do_interp_parts_matrix.shape[1] and \
                           do_interp_parts_matrix[iter_link_idx, iter_part_idx] == 1 and \
                           verts_part_visible.shape[0] > 3 and iter_part_idx < len(interp_resolution_parts): 
                            min_coords_xy = np.min(verts_part_visible[:, :2], axis=0)
                            max_coords_xy = np.max(verts_part_visible[:, :2], axis=0)
                            res_val = interp_resolution_parts[iter_part_idx]
                            if max_coords_xy[0] > min_coords_xy[0] and max_coords_xy[1] > min_coords_xy[1] and res_val > 0:
                                xq_coords = np.arange(min_coords_xy[0], max_coords_xy[0] + res_val, res_val) # include endpoint for arange
                                yq_coords = np.arange(min_coords_xy[1], max_coords_xy[1] + res_val, res_val)
                                if len(xq_coords) > 1 and len(yq_coords) > 1:
                                    xq_grid, yq_grid = np.meshgrid(xq_coords, yq_coords)
                                    try:
                                        zq_grid = scipy.interpolate.griddata(
                                            verts_part_visible[:, :2],
                                            verts_part_visible[:, 2],
                                            (xq_grid, yq_grid), method='cubic', fill_value=np.nan
                                        )
                                        valid_interp_mask = ~np.isnan(zq_grid)
                                        if np.any(valid_interp_mask):
                                            new_interp_points = np.vstack([
                                                xq_grid[valid_interp_mask].flatten(),
                                                yq_grid[valid_interp_mask].flatten(),
                                                zq_grid[valid_interp_mask].flatten()
                                            ]).T
                                            verts_part_visible = np.vstack([verts_part_visible, new_interp_points])
                                    except Exception as e_interp:
                                        print(f"Griddata interpolation failed: {e_interp}")
                        
                        verts_for_summation_curr_t = np.vstack([verts_for_summation_curr_t, verts_part_visible])
                        scale_body_surface_for_summation_curr_t = np.concatenate([
                            scale_body_surface_for_summation_curr_t,
                            scaling_body[iter_part_idx] * np.ones(verts_part_visible.shape[0])])
                        beamwidth_for_summation_curr_t = np.concatenate([
                            beamwidth_for_summation_curr_t,
                            beamwidth_body_default * np.ones(verts_part_visible.shape[0])])
                else:
                    verts_curr_noarm_at_t = verts_curr_all_parts_at_t[set_noarm, :]
                    if verts_curr_noarm_at_t.shape[0] > 0:
                        idx_visible_tx_noarm = hpr_equivalent(verts_curr_noarm_at_t, Tx_pos.reshape(1,-1), 2.25)
                        idx_visible_rx_noarm = hpr_equivalent(verts_curr_noarm_at_t, Rx_pos.reshape(1,-1), 2.25)
                        idx_visible_noarm_final = np.intersect1d(idx_visible_tx_noarm, idx_visible_rx_noarm)
                        verts_for_summation_curr_t = verts_curr_noarm_at_t[idx_visible_noarm_final, :]
                    scale_body_surface_for_summation_curr_t = np.ones(verts_for_summation_curr_t.shape[0])
                    beamwidth_for_summation_curr_t = beamwidth_body_default * np.ones(verts_for_summation_curr_t.shape[0])

                if verts_for_summation_curr_t.shape[0] == 0:
                    RS_signal[i_time_step] = np.exp(1j * 2 * np.pi * dist_Tx_Rx / lambda_wave) / (4 * np.pi * (dist_Tx_Rx + epsilon_dist))
                    continue
                
                normals = pcnormals_open3d(verts_for_summation_curr_t, k_neighbors_if_no_radius=12)
                point_Tx_vector = Tx_pos - verts_for_summation_curr_t
                point_Rx_vector = Rx_pos - verts_for_summation_curr_t
                point_Rx_vector_norm = np.linalg.norm(point_Rx_vector, axis=1, keepdims=True)
                point_Rx_vector_unit = point_Rx_vector / (point_Rx_vector_norm + epsilon_dist)
                incident_vector = -point_Tx_vector 
                incident_vector_norm = np.linalg.norm(incident_vector, axis=1, keepdims=True)
                incident_vector_unit = incident_vector / (incident_vector_norm + epsilon_dist)
                dot_inc_normal = np.sum(incident_vector_unit * normals, axis=1, keepdims=True)
                ref_vector_unit = incident_vector_unit - 2 * dot_inc_normal * normals
                dot_rx_ref = np.sum(point_Rx_vector_unit * ref_vector_unit, axis=1)
                dot_rx_ref = np.clip(dot_rx_ref, -1.0, 1.0)
                angle_deg = np.rad2deg(np.arccos(dot_rx_ref))
                scale_ref_beam = np.exp(-angle_deg**2 / (2 * beamwidth_for_summation_curr_t**2))
                dist_path = np.linalg.norm(point_Tx_vector, axis=1) + np.linalg.norm(point_Rx_vector, axis=1)
                summand_complex_exp = np.exp(1j * 2 * np.pi * dist_path / lambda_wave)
                term1_summand = (scale_body_surface_for_summation_curr_t * scale_ref_beam * summand_complex_exp / (4 * np.pi * (dist_path + epsilon_dist)))
                RS_signal[i_time_step] = np.sum(term1_summand) + np.exp(1j * 2 * np.pi * dist_Tx_Rx / lambda_wave) / (4 * np.pi * (dist_Tx_Rx + epsilon_dist))

            RS_all_py[iter_link_idx] = RS_signal
            
            print('Generating spectrogram...')
            actual_ts_spec = ts_interp
            if tq_interp.size > 1: actual_ts_spec = tq_interp[1] - tq_interp[0]
            elif t_times.size > 1: actual_ts_spec = t_times[1] - t_times[0]
            if actual_ts_spec <= 0: print(f"Invalid time step for spec. link {iter_link_idx}"); continue

            window_duration_sec = 0.4
            window_samples_spec = int(window_duration_sec / actual_ts_spec)
            fs_spec = 1.0 / actual_ts_spec
            if not (0 < window_samples_spec <= len(RS_signal)): print(f"Invalid window for spec. {window_samples_spec} vs {len(RS_signal)}"); continue

            abs_RS_sq_signal = np.abs(RS_signal)**2
            window_len_rlf_samples = int(0.15 / actual_ts_spec) if actual_ts_spec > 0 else 1
            if window_len_rlf_samples < 1: window_len_rlf_samples = 1
            if len(abs_RS_sq_signal) > window_len_rlf_samples :
                 abs_RS_filtered_signal = remove_low_freq(abs_RS_sq_signal, window_len_rlf_samples)
            else: abs_RS_filtered_signal = abs_RS_sq_signal

            freq_axis_spec = np.arange(1, 101, dtype=float)
            noverlap_spec = max(0, window_samples_spec - 1)

            f_calc_s, t_calc_s, Sxx_calc = scipy.signal.spectrogram(abs_RS_filtered_signal, fs=fs_spec, window='hamming', nperseg=window_samples_spec, noverlap=noverlap_spec, nfft=max(256, window_samples_spec), scaling='density')
            
            if Sxx_calc.size > 0 and f_calc_s.size > 0:
                interp_Sxx_func = scipy.interpolate.interp1d(f_calc_s, np.abs(Sxx_calc), axis=0, kind='linear', fill_value=0.0, bounds_error=False)
                sp_all_py[iter_link_idx] = interp_Sxx_func(freq_axis_spec)
                T_s_py_glob, F_s_py_glob = t_calc_s, freq_axis_spec # Store T,F from last successful link

        if plt.get_fignums(): plt.close('all') 
        fig_spec_plot, axes_spec_plot = plt.subplots(3, 1, figsize=(10, 12), sharex=True, squeeze=False)
        for iter_link_plot_idx in active_links_for_sim:
            ax_curr_plot = axes_spec_plot[iter_link_plot_idx, 0]
            video_spectrogram_to_plot = sp_all_py[iter_link_plot_idx]
            link_name_plot = name_link[iter_link_plot_idx]
            ax_curr_plot.set_title(f'Action: {activity_id} ({cls_name}), Vid: {current_video_name_part}, Link: {link_name_plot}')
            ax_curr_plot.set_xlabel('time (s)'); ax_curr_plot.set_ylabel('freq (Hz)')
            if video_spectrogram_to_plot.size == 0 or T_s_py_glob.size == 0 or F_s_py_glob.size == 0:
                ax_curr_plot.text(0.5, 0.5, 'No data', ha='center', va='center'); ax_curr_plot.axis('off'); continue
            video_spectrogram_g_plot = scipy.ndimage.gaussian_filter(video_spectrogram_to_plot, sigma=2)
            ax_curr_plot.pcolormesh(T_s_py_glob, F_s_py_glob, video_spectrogram_g_plot, shading='gouraud', cmap='jet', vmin=np.min(video_spectrogram_g_plot), vmax=np.max(video_spectrogram_g_plot))
            ax_curr_plot.axis([T_s_py_glob[0], T_s_py_glob[-1], F_s_py_glob[0], F_s_py_glob[-1]])
        plt.tight_layout()
        fig_save_path = os.path.join(script_dir, f'spectrogram_activity{activity_id}_vid{current_video_name_part}.png')
        try: fig_spec_plot.savefig(fig_save_path); print(f"Saved spectrogram plot: {fig_save_path}"); plt.close(fig_spec_plot)
        except Exception as e: print(f"Error saving spectrogram fig: {e}")

        folder_save_base_path = os.path.join(script_dir, 'simulated_spectrograms')
        folder_save_cls_path = os.path.join(folder_save_base_path, cls_name)
        os.makedirs(folder_save_cls_path, exist_ok=True)
        save_mat_file_path = os.path.join(folder_save_cls_path, f'vid-{current_video_name_part}.mat')
        sp_all_to_save = [item if isinstance(item, np.ndarray) else np.array([]) for item in sp_all_py]
        data_to_save_dict = {'sp_all': np.array(sp_all_to_save, dtype=object), 'T': T_s_py_glob, 'F': F_s_py_glob, 'period_start_times': period_start_times_vals}
        try: scipy.io.savemat(save_mat_file_path, data_to_save_dict); print(f"Saved data to {save_mat_file_path}")
        except Exception as e: print(f"Error saving .mat data: {e}")

if __name__ == '__main__':
    main()
    if plt.get_fignums(): plt.close('all')

