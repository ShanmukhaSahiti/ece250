import numpy as np

def get_original_pt_location(x, person_box, cams, im_shape):
    """
    This function takes a point x in the bounding box and returns its location
    in the original video frame.

    Args:
        x (np.ndarray): Input points, expected to be N x 3.
        person_box (list or np.ndarray): Bounding box [y_start, x_start, y_end, x_end]
                                         (using typical image processing order: row_start, col_start, row_end, col_end).
                                         The Matlab code uses [person_box(2), person_box(1), person_box(4), person_box(3)]
                                         which seems to map to [x_start, y_start, x_end, y_end] if person_box is [ymin, xmin, ymax, xmax].
                                         Let's assume person_box is [y_min_cv, x_min_cv, y_max_cv, x_max_cv]
                                         Matlab: person_box(1)=ymin, person_box(2)=xmin, person_box(3)=ymax, person_box(4)=xmax
                                         So, x_min_matlab = person_box[1], y_min_matlab = person_box[0]
                                         x_max_matlab = person_box[3], y_max_matlab = person_box[2]
        cams (list or np.ndarray): Camera parameters, e.g., [cx, cy, cz_offset_or_scale].
                                    Matlab: cams(2) and cams(3). Assuming Python cams will be [cams_1, cams_2, cams_3]
                                    So cams[1] in Python maps to cams(2) in Matlab.
                                    And cams[2] in Python maps to cams(3) in Matlab.
        im_shape (tuple or list): Shape of the image (height, width).
                                  Matlab im_shape(1) is height, im_shape(2) is width.

    Returns:
        np.ndarray: Transformed points (N x 2), [x_transformed, y_transformed].
    """
    # Ensure x is a NumPy array
    x_transformed = np.array(x, dtype=float)
    if x_transformed.ndim == 1: # If a single point is passed
        x_transformed = x_transformed.reshape(1, -1)

    margin_expand = 40

    # Matlab: x_min = max([person_box(2)-margin_expand, 0]);
    # person_box in Matlab: [ymin, xmin, ymax, xmax]
    # So, person_box(2) is xmin
    # Python: Assuming person_box is [ymin_cv, xmin_cv, ymax_cv, xmax_cv] (common OpenCV/Python ordering)
    # Let's clarify the person_box indexing.
    # The original Matlab script uses `person_box` from `load(['./MeshInfoMatFiles/' video_name, '/frame_',sprintf('%04d',frame_idx),'.mat']);`
    # And then `SMPL_params.person_box`. We need to check how `person_box` is structured in those .mat files.
    # For now, let's assume the direct translation of indices:
    # person_box[0] -> matlab person_box(1) (ymin)
    # person_box[1] -> matlab person_box(2) (xmin)
    # person_box[2] -> matlab person_box(3) (ymax)
    # person_box[3] -> matlab person_box(4) (xmax)

    # x_min in Matlab refers to the column start
    # y_min in Matlab refers to the row start
    
    # Matlab: x_min_bbox = person_box(2)
    # Matlab: y_min_bbox = person_box(1)
    # Matlab: x_max_bbox = person_box(4)
    # Matlab: y_max_bbox = person_box(3)

    pb_y_min, pb_x_min, pb_y_max, pb_x_max = person_box[0], person_box[1], person_box[2], person_box[3]

    # x_min_calc corresponds to the minimum column index for the expanded box
    x_min_calc = max(pb_x_min - margin_expand, 0)
    # y_min_calc corresponds to the minimum row index for the expanded box
    y_min_calc = max(pb_y_min - margin_expand, 0)
    
    # x_max_calc corresponds to the maximum column index for the expanded box
    # im_shape(2) is width. In Python, im_shape[1] is width.
    x_max_calc = min(pb_x_max + margin_expand, im_shape[1] - 1)
    # y_max_calc corresponds to the maximum row index for the expanded box
    # im_shape(1) is height. In Python, im_shape[0] is height.
    y_max_calc = min(pb_y_max + margin_expand, im_shape[0] - 1)

    height_expanded_box = y_max_calc - y_min_calc
    width_expanded_box = x_max_calc - x_min_calc

    largest_dim = max(height_expanded_box, width_expanded_box)
    smallest_dim = min(height_expanded_box, width_expanded_box)

    # x(:,1) = x(:,1) + cams(2);
    # x(:,3) = -x(:,3) + cams(3);
    # Assuming cams is 0-indexed in Python and corresponds to Matlab's 1-indexed access
    # cams[1] in Python maps to cams(2) in Matlab
    # cams[2] in Python maps to cams(3) in Matlab
    x_transformed[:, 0] = x_transformed[:, 0] + cams[1]
    x_transformed[:, 2] = -x_transformed[:, 2] + cams[2]
    
    x_transformed = x_transformed * (largest_dim / 2.0)

    # if y_max - y_min >= x_max - x_min  (i.e. height_expanded_box >= width_expanded_box)
    if height_expanded_box >= width_expanded_box:
        # x(:,3) = x(:,3) + y_min + largest_dim/2;  (y-coordinate in image)
        x_transformed[:, 2] = x_transformed[:, 2] + y_min_calc + largest_dim / 2.0
        # x(:,1) = x(:,1) + x_min + smallest_dim/2; (x-coordinate in image)
        x_transformed[:, 0] = x_transformed[:, 0] + x_min_calc + smallest_dim / 2.0
    else:
        # x(:,1) = x(:,1) + x_min + largest_dim/2; (x-coordinate in image)
        x_transformed[:, 0] = x_transformed[:, 0] + x_min_calc + largest_dim / 2.0
        # x(:,3) = x(:,3) + y_min + smallest_dim/2; (y-coordinate in image)
        x_transformed[:, 2] = x_transformed[:, 2] + y_min_calc + smallest_dim / 2.0
        
    # y = [x(:,1), x(:,3)];
    # This means the output should be [transformed_x_column, transformed_y_column]
    # The Matlab code effectively maps original x's 1st column to image x and 3rd column to image y.
    y_output = x_transformed[:, [0, 2]] # Select 1st and 3rd columns

    return y_output

if __name__ == '__main__':
    # Example Usage (mirroring potential interpretation of Matlab script)
    
    # Sample input point (e.g., from SMPL model, centered around origin)
    # x is N x 3
    sample_x = np.array([[0.1, 0.2, 0.3],
                         [-0.1, -0.2, -0.3]]) 
                         
    # Person bounding box from .mat file (e.g. SMPL_params.person_box)
    # Matlab format: [ymin, xmin, ymax, xmax]
    # Let's assume this is what's loaded from the .mat file directly.
    person_box_matlab_style = np.array([100, 50, 300, 250]) # y_min=100, x_min=50, y_max=300, x_max=250

    # Camera parameters (e.g., SMPL_params.cams)
    # Matlab: cams = [scale, trans_x, trans_y] based on typical SMPL usage.
    # The function uses cams(2) and cams(3).
    # If cams = [s, tx, ty], then cams(2) is tx, cams(3) is ty.
    # Python equivalent: cams_py = [s, tx, ty]
    # cams_py[1] would be tx, cams_py[2] would be ty.
    sample_cams = np.array([0.8, 0.05, 0.1]) # [scale_factor, translation_x_offset, translation_y_offset_for_z_axis]

    # Image shape (height, width)
    sample_im_shape = (480, 640) # height=480, width=640

    print(f"Input x:\n{sample_x}")
    print(f"Person box (Matlab-style ymin, xmin, ymax, xmax): {person_box_matlab_style}")
    print(f"Cams (s, tx, ty): {sample_cams}")
    print(f"Image_shape (height, width): {sample_im_shape}")

    transformed_points = get_original_pt_location(sample_x.copy(), person_box_matlab_style, sample_cams, sample_im_shape)
    print(f"\nTransformed points (image x, image y):\n{transformed_points}")

    # Test case from a potential scenario
    # Assume point is at origin in its local coordinate system before cam transformation
    x_origin = np.array([[0.0, 0.0, 0.0]])
    # Box is [ymin=10, xmin=20, ymax=110, xmax=120] -> height=100, width=100
    # So largest_dim = 100, smallest_dim = 100
    # Expanded box with margin 40:
    # y_min_calc = max(10-40,0) = 0
    # x_min_calc = max(20-40,0) = 0
    # y_max_calc = min(110+40, 480-1) = min(150, 479) = 150
    # x_max_calc = min(120+40, 640-1) = min(160, 639) = 160
    # height_expanded_box = 150 - 0 = 150
    # width_expanded_box = 160 - 0 = 160
    # largest_dim = 160, smallest_dim = 150 (Error in manual calc, width_expanded_box is larger)
    # Recalculate:
    # pb_y_min=10, pb_x_min=20, pb_y_max=110, pb_x_max=120
    # im_shape = (480, 640)
    # margin_expand = 40
    # x_min_calc = max(20 - 40, 0) = 0
    # y_min_calc = max(10 - 40, 0) = 0
    # x_max_calc = min(120 + 40, 640 - 1) = min(160, 639) = 160
    # y_max_calc = min(110 + 40, 480 - 1) = min(150, 479) = 150
    # height_expanded_box = 150 - 0 = 150
    # width_expanded_box  = 160 - 0 = 160
    # largest_dim = 160
    # smallest_dim = 150
    
    # Let's use a simple box: person_box_test = [0,0,100,100], im_shape_test = (200,200)
    # margin_expand = 0
    # x_min_calc = 0, y_min_calc = 0, x_max_calc = 99, y_max_calc = 99
    # height = 99, width = 99. largest_dim = 99, smallest_dim = 99
    # cams_test = [1, 0, 0] (s=1, tx=0, ty=0)
    # x_transformed = x_origin.copy() # [[0,0,0]]
    # x_transformed[:,0] += cams_test[1] => x_transformed[:,0] += 0 => [[0,0,0]]
    # x_transformed[:,2] = -x_transformed[:,2] + cams_test[2] => -0+0=0 => [[0,0,0]]
    # x_transformed = x_transformed * (largest_dim / 2.0) => [[0,0,0]] * (99/2) = [[0,0,0]]
    # height_expanded_box (99) >= width_expanded_box (99) -> True
    # x_transformed[:,2] = x_transformed[:,2] + y_min_calc + largest_dim/2.0
    #                    = 0 + 0 + 99/2 = 49.5
    # x_transformed[:,0] = x_transformed[:,0] + x_min_calc + smallest_dim/2.0
    #                    = 0 + 0 + 99/2 = 49.5
    # y_output = [[49.5, 49.5]]
    # This puts the origin (0,0,0) at the center of the box [0,0,99,99] -> (49.5, 49.5) which seems correct.

    # Test with a non-zero cam translation
    # cams_test2 = [1, 0.1, 0.2] (s=1, tx=0.1, ty=0.2)
    # x_transformed = x_origin.copy() # [[0,0,0]]
    # x_transformed[:,0] += cams_test2[1] => x_transformed[:,0] += 0.1 => [[0.1,0,0]]
    # x_transformed[:,2] = -x_transformed[:,2] + cams_test2[2] => -0 + 0.2 = 0.2 => [[0.1,0,0.2]]
    # x_transformed = x_transformed * (largest_dim / 2.0) => [[0.1,0,0.2]] * 49.5 = [[4.95, 0, 9.9]]
    # height_expanded_box (99) >= width_expanded_box (99) -> True
    # x_transformed[:,2] = x_transformed[:,2] + y_min_calc + largest_dim/2.0
    #                    = 9.9 + 0 + 49.5 = 59.4
    # x_transformed[:,0] = x_transformed[:,0] + x_min_calc + smallest_dim/2.0
    #                    = 4.95 + 0 + 49.5 = 54.45
    # y_output = [[54.45, 59.4]]

    person_box_test = np.array([0, 0, 100, 100]) # ymin, xmin, ymax, xmax (so height 100, width 100)
    im_shape_test = (200, 200)                  # image is 200x200
                                                # Matlab: person_box(1)=0, (2)=0, (3)=100, (4)=100
                                                # im_shape(1)=200, im_shape(2)=200
                                                # x_min_matlab = 0, y_min_matlab = 0, x_max_matlab=100, y_max_matlab=100
                                                
                                                # Python: pb_y_min=0, pb_x_min=0, pb_y_max=100, pb_x_max=100
                                                # im_shape[0]=200, im_shape[1]=200

    cams_test = np.array([1.0, 0.0, 0.0])       # s, tx, ty
    x_point_test = np.array([[0.0, 0.0, 0.0]])

    print("\n--- Test Case 1: Origin point, no cam translation, square box at origin ---")
    print(f"Input x: {x_point_test}")
    print(f"Person box: {person_box_test}")
    print(f"Cams: {cams_test}")
    print(f"Image shape: {im_shape_test}")
    # Expected output: center of the box. Box is 0,0 to 100,100. Center should be roughly 50,50 (or 49.5, 49.5 due to 0-indexing and dim/2)
    # margin_expand = 0 for simplicity in manual trace.
    # x_min_calc = max(0-0,0)=0
    # y_min_calc = max(0-0,0)=0
    # x_max_calc = min(100+0, 200-1)=100
    # y_max_calc = min(100+0, 200-1)=100
    # height_exp = 100, width_exp = 100. largest_dim=100, smallest_dim=100
    # x_trans = [[0,0,0]]
    # x_trans[:,0] += 0 -> 0
    # x_trans[:,2] = -0 + 0 -> 0
    # x_trans = [[0,0,0]] * (100/2) = [[0,0,0]]
    # if 100 >= 100:
    #   x_trans[:,2] = 0 + 0 + 100/2 = 50
    #   x_trans[:,0] = 0 + 0 + 100/2 = 50
    # result: [[50,50]]
    
    # With margin_expand = 40
    # x_min_calc = max(0 - 40, 0) = 0
    # y_min_calc = max(0 - 40, 0) = 0
    # x_max_calc = min(100 + 40, 200 - 1) = min(140, 199) = 140
    # y_max_calc = min(100 + 40, 200 - 1) = min(140, 199) = 140
    # height_expanded_box = 140
    # width_expanded_box = 140
    # largest_dim = 140
    # smallest_dim = 140
    # x_t = [[0,0,0]]
    # x_t[:,0] += 0 = 0
    # x_t[:,2] = -0+0 = 0
    # x_t = [[0,0,0]] * (140/2) = [[0,0,0]]
    # if 140 >= 140:
    #   x_t[:,2] = 0 + y_min_calc(0) + largest_dim(140)/2 = 70
    #   x_t[:,0] = 0 + x_min_calc(0) + smallest_dim(140)/2 = 70
    # result [[70,70]] -> This is the center of the *expanded* box [0,0,140,140]

    output_test1 = get_original_pt_location(x_point_test.copy(), person_box_test, cams_test, im_shape_test)
    print(f"Output test 1: {output_test1}") # Expected with margin 40: [[70,70]]


    cams_test2 = np.array([1.0, 0.1, 0.2]) # s, tx=0.1, ty=0.2
    print("\n--- Test Case 2: Origin point, with cam translation ---")
    print(f"Input x: {x_point_test}")
    print(f"Person box: {person_box_test}")
    print(f"Cams: {cams_test2}")
    print(f"Image shape: {im_shape_test}")
    # x_t = [[0,0,0]]
    # x_t[:,0] += 0.1 => 0.1
    # x_t[:,2] = -0 + 0.2 => 0.2
    # x_t = [[0.1, 0, 0.2]]
    # largest_dim = 140 (from previous calculation with margin 40)
    # x_t = [[0.1,0,0.2]] * (140/2) = [[0.1,0,0.2]] * 70 = [[7, 0, 14]]
    # if 140 >= 140:
    #   x_t[:,2] = 14 + y_min_calc(0) + largest_dim(140)/2 = 14 + 0 + 70 = 84
    #   x_t[:,0] = 7  + x_min_calc(0) + smallest_dim(140)/2 = 7  + 0 + 70 = 77
    # result [[77, 84]]
    output_test2 = get_original_pt_location(x_point_test.copy(), person_box_test, cams_test2, im_shape_test)
    print(f"Output test 2: {output_test2}") # Expected [[77,84]]

    # Test with different aspect ratio for the expanded box
    person_box_test3 = np.array([0, 0, 50, 100]) # ymin, xmin, ymax, xmax (height 50, width 100)
    im_shape_test3 = (200, 200)
    cams_test3 = np.array([1.0, 0.0, 0.0])
    print("\n--- Test Case 3: Origin point, no cam trans, landscape box ---")
    print(f"Input x: {x_point_test}")
    print(f"Person box: {person_box_test3}")
    print(f"Cams: {cams_test3}")
    print(f"Image shape: {im_shape_test3}")
    # margin_expand = 40
    # pb_y_min=0, pb_x_min=0, pb_y_max=50, pb_x_max=100
    # x_min_calc = max(0-40,0) = 0
    # y_min_calc = max(0-40,0) = 0
    # x_max_calc = min(100+40, 199) = 140
    # y_max_calc = min(50+40, 199) = 90
    # height_expanded_box = 90 - 0 = 90
    # width_expanded_box  = 140 - 0 = 140
    # largest_dim = 140 (width)
    # smallest_dim = 90 (height)
    # x_t = [[0,0,0]]
    # x_t[:,0] += 0 => 0
    # x_t[:,2] = -0+0 => 0
    # x_t = [[0,0,0]] * (largest_dim(140)/2) = [[0,0,0]] * 70 = [[0,0,0]]
    # if height_expanded_box (90) >= width_expanded_box (140) -> False. Else branch:
    #   x_t[:,0] = 0 + x_min_calc(0) + largest_dim(140)/2.0 = 0 + 0 + 70 = 70
    #   x_t[:,1] -> x_transformed[:, 2] (Original typo in comment, should be x_transformed[:, 2])
    #   x_t[:,2] = 0 + y_min_calc(0) + smallest_dim(90)/2.0 = 0 + 0 + 45 = 45
    # result [[70,45]]
    output_test3 = get_original_pt_location(x_point_test.copy(), person_box_test3, cams_test3, im_shape_test3)
    print(f"Output test 3: {output_test3}") # Expected [[70,45]]
    
    # Check a point not at origin
    x_point_test4 = np.array([[0.1, 0.0, 0.1]])
    print("\n--- Test Case 4: Non-Origin point, no cam trans, landscape box ---")
    print(f"Input x: {x_point_test4}")
    # (Using box and im_shape from Test Case 3)
    # largest_dim = 140, smallest_dim = 90
    # x_min_calc = 0, y_min_calc = 0
    # x_t = [[0.1,0,0.1]]
    # x_t[:,0] += 0 => 0.1
    # x_t[:,2] = -0.1+0 => -0.1
    # x_t = [[0.1,0,-0.1]]
    # x_t = [[0.1,0,-0.1]] * (140/2) = [[0.1,0,-0.1]] * 70 = [[7,0,-7]]
    # if height_expanded_box (90) >= width_expanded_box (140) -> False. Else branch:
    #   x_t[:,0] = 7 + x_min_calc(0) + largest_dim(140)/2.0 = 7 + 0 + 70 = 77
    #   x_t[:,2] = -7 + y_min_calc(0) + smallest_dim(90)/2.0 = -7 + 0 + 45 = 38
    # result [[77,38]]
    output_test4 = get_original_pt_location(x_point_test4.copy(), person_box_test3, cams_test3, im_shape_test3)
    print(f"Output test 4: {output_test4}") # Expected [[77,38]]

    # Ensure it works for multiple points
    x_points_test5 = np.array([[0.0,0.0,0.0], [0.1,0.0,0.1]])
    print("\n--- Test Case 5: Multiple points ---")
    # Expected: [[70,45], [77,38]]
    output_test5 = get_original_pt_location(x_points_test5.copy(), person_box_test3, cams_test3, im_shape_test3)
    print(f"Output test 5: {output_test5}")

    # Test with original sample data from the top of the main block
    print("\n--- Test Case 6: Original sample data ---")
    print(f"Input x:\n{sample_x}")
    print(f"Person box (Matlab-style ymin, xmin, ymax, xmax): {person_box_matlab_style}") # [100, 50, 300, 250]
    print(f"Cams (s, tx, ty): {sample_cams}") # [0.8, 0.05, 0.1]
    print(f"Image_shape (height, width): {sample_im_shape}") # (480, 640)
    # pb_y_min=100, pb_x_min=50, pb_y_max=300, pb_x_max=250
    # margin_expand = 40
    # x_min_calc = max(50 - 40, 0) = 10
    # y_min_calc = max(100 - 40, 0) = 60
    # x_max_calc = min(250 + 40, 640 - 1) = min(290, 639) = 290
    # y_max_calc = min(300 + 40, 480 - 1) = min(340, 479) = 340
    # height_expanded_box = 340 - 60 = 280
    # width_expanded_box  = 290 - 10 = 280
    # largest_dim = 280
    # smallest_dim = 280

    # sample_x = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    # sample_cams = np.array([0.8, 0.05, 0.1]) -> use cams[1]=0.05, cams[2]=0.1

    # Point 1: [0.1, 0.2, 0.3]
    # x_t1 = [0.1, 0.2, 0.3]
    # x_t1[0] += 0.05 => 0.15
    # x_t1[2] = -0.3 + 0.1 => -0.2
    # x_t1_after_cam = [0.15, 0.2, -0.2]
    # x_t1_scaled = [0.15, 0.2, -0.2] * (largest_dim(280)/2.0) = [0.15, 0.2, -0.2] * 140
    #             = [21, 28, -28]
    # if height_expanded_box (280) >= width_expanded_box (280) -> True
    #   x_t1_final_z = -28 + y_min_calc(60) + largest_dim(280)/2.0 = -28 + 60 + 140 = 172
    #   x_t1_final_x = 21  + x_min_calc(10) + smallest_dim(280)/2.0 = 21 + 10 + 140 = 171
    # Result P1: [171, 172]

    # Point 2: [-0.1, -0.2, -0.3]
    # x_t2 = [-0.1, -0.2, -0.3]
    # x_t2[0] += 0.05 => -0.05
    # x_t2[2] = -(-0.3) + 0.1 => 0.3 + 0.1 = 0.4
    # x_t2_after_cam = [-0.05, -0.2, 0.4]
    # x_t2_scaled = [-0.05, -0.2, 0.4] * 140
    #             = [-7, -28, 56]
    # if height_expanded_box (280) >= width_expanded_box (280) -> True
    #   x_t2_final_z = 56 + y_min_calc(60) + largest_dim(280)/2.0 = 56 + 60 + 140 = 256
    #   x_t2_final_x = -7  + x_min_calc(10) + smallest_dim(280)/2.0 = -7 + 10 + 140 = 143
    # Result P2: [143, 256]
    # Expected: [[171, 172], [143, 256]]
    transformed_points_sample = get_original_pt_location(sample_x.copy(), person_box_matlab_style, sample_cams, sample_im_shape)
    print(f"Transformed points (sample case):\n{transformed_points_sample}")

    # A quick check on indexing of person_box and cams in main script `generate_wifi_from_video.py`
    # `person_box = frame_data['SMPL_params']['person_box'][0]`
    # `cams = frame_data['SMPL_params']['cam_params'][0]`
    # This suggests that `person_box` and `cams` might be 2D arrays in the .mat file (e.g., 1x4 and 1x3) and are being
    # correctly selected as 1D arrays.
    # The structure of `person_box` in Matlab `[ymin, xmin, ymax, xmax]` seems to be standard.
    # My Python `person_box` argument as `[pb_y_min, pb_x_min, pb_y_max, pb_x_max]` matches this.
    # For `cams`, Matlab code uses `cams(2)` and `cams(3)`. If `cams` in Matlab is `[s, tx, ty]`, then
    # `cams(1)=s`, `cams(2)=tx`, `cams(3)=ty`.
    # My Python `cams` argument as `[s, tx, ty]` means Python `cams[0]=s`, `cams[1]=tx`, `cams[2]=ty`.
    # So, `x_transformed[:, 0] = x_transformed[:, 0] + cams[1]` (uses tx)
    # `x_transformed[:, 2] = -x_transformed[:, 2] + cams[2]` (uses ty)
    # This mapping seems correct. 