function [verts, joints, period_start_frames] = mesh_alignment_algorithm(all_verts_time, all_joints_time,...
    person_box_time, person_mask_time, cams_time, im_shape,...
    cls, cls_category, vid)
% MESH ALIGNMENT ALGORITHM  implements the mesh alignment algorithm via
% eigen analysis as in sec 3.2 (or 4.2.2) of the paper.
%   INPUTS:
%	all_verts_time = coordinates of all mesh points with time before alignment
%   all_joints_time = coordinates of all joints with time before alignment
%   person_box_time = the bounding box of the person in the video frames
%   person_mask_time = the binary mask of the person in the video frames
%   cams_time = camera parameter in all frames
%   im_shape = dimensions of the video frames in pixels
%   cls = the activity to be adjusted
%   cls_category = the category of the activity (some activities require
%       same alignment so they get the same category number).
%   vid = which video of the activity
%   OUTPUTS:
%   verts = coordinates of all mesh points with time after alignment
%   joints = coordinates of all joints with time after alignment
%   period_start_frames = the frames that indicate the start of each period
%       in the periodic activities (NaN for non-periodic activities)
% COPYRIGHT (c): H. Cai, B. Korany, and C. Karanam (UCSB, 2020)


% load the indeces of different body parts in the meshes
load torso_vts_idx
load left_toes_idx
load right_toes_idx
load right_fingers_idx
load left_fingers_idx
load leg1_idx
load leg2_idx
load left_foot_idx
load right_foot_idx
load('left_thigh_idx.mat');
load('right_thigh_idx.mat');
load('hip_idx.mat')
front_torso_idx = 3051;
back_torso_idx = 461;

% joints
% 1. ankle left
% 2. knee left
% 3. hip left
% 4. hip right
% 5. knee right
% 6. ankle right
% 7. wrist left
% 8. elbow left
% 9. shoulder left
% 10. shoulder right
% 11. elbow right
% 12. wrist right
% 13. chest/neck
% 14. forehead
% 15. nose
% 16. left eye
% 17. right eye
% 18. left ear
% 19. right ear



num_frames = size(all_verts_time,3); %total number of frames in video

%% ========================================================================
%% standing actions w/ one foot/both feet static, applies to forward lunge,
%% front leg raise, lateral lunge, and stiff-leg deadlift
%% ========================================================================
if cls_category == 1 || cls_category == 2 %|| cls_category == 3
    verts=all_verts_time; 
    joints = all_joints_time; 
    
    % Getting the reference frame (the frame in which the person is
    % standing
    person_box_height = person_box_time(3,:)-person_box_time(1,:);
    person_box_width = person_box_time(4,:)-person_box_time(2,:);
    if cls_category == 1
        [~, standing_frame] = max(person_box_height);
        
        series_cat1 = person_box_height;
        N_cat1 = length(series_cat1);
        disp(['DEBUG: cls_category=1: series length = ', num2str(N_cat1)]);
        person_box_ACF = []; % Initialize
        if N_cat1 >= 2 % Must have at least 2 points for any ACF
            lags_val_cat1 = N_cat1 - 50; % Desired
            disp(['DEBUG: cls_category=1: desired lags (-50) = ', num2str(lags_val_cat1)]);
            if lags_val_cat1 > 0 && lags_val_cat1 < N_cat1 
                person_box_ACF = autocorr(series_cat1, 'NumLags', lags_val_cat1);
                disp('DEBUG: cls_category=1: autocorr with -50 lags SUCCEEDED.');
            else 
                lags_fallback_cat1 = N_cat1 - 1;
                disp(['DEBUG: cls_category=1: -50 lags invalid, trying fallback N-1 lags = ', num2str(lags_fallback_cat1)]);
                if lags_fallback_cat1 > 0 % Ensure N-1 is at least 1
                    person_box_ACF = autocorr(series_cat1, 'NumLags', lags_fallback_cat1);
                    disp('DEBUG: cls_category=1: autocorr with N-1 lags SUCCEEDED.');
                else
                    disp('DEBUG: cls_category=1: N-1 lags also invalid (<=0). ACF set to [].');
                end
            end
        else
            disp('DEBUG: cls_category=1: Series too short for ACF. ACF set to [].');
        end
        if ~isempty(person_box_ACF)
             disp(['DEBUG: cls_category=1: Final ACF class: ', class(person_box_ACF), ', size: [', num2str(size(person_box_ACF,1)), ' ', num2str(size(person_box_ACF,2)), ']']);
        end

        min_peak_dist = floor(length(person_box_ACF) / 2) - 1;
        if min_peak_dist < 1
            min_peak_dist = 1;
        end
        [~,peak_locs] = findpeaks(person_box_ACF,'MinPeakDistance',min_peak_dist);
        
        if isempty(peak_locs)
            action_period = NaN; % No peaks found, so period is undefined or use a default
            disp('DEBUG: No peaks found by findpeaks. action_period set to NaN.');
        else
            peak_locs = peak_locs(:)'; % Ensure peak_locs is a row vector for [0 peak_locs]
            action_period = mean(diff([0 peak_locs]));
            disp(['DEBUG: Peaks found. action_period calculated as: ', num2str(action_period)]);
        end
    else
        [~, standing_frame] = max(person_box_height- person_box_width);

        debug_series_data = person_box_height - person_box_width;
        disp('DEBUG: About to call autocorr (cls_category not 1). Series data properties:');
        disp(['  Class: ', class(debug_series_data)]);
        disp(['  Size: [', num2str(size(debug_series_data,1)), ' ', num2str(size(debug_series_data,2)), ']']);
        disp(['  Is all finite? ', num2str(all(isfinite(debug_series_data)))]');
        disp(['  Is real? ', num2str(isreal(debug_series_data))]);
        
        person_box_ACF = []; % Initialize
        try
            disp('DEBUG: Trying autocorr in TRY block for cls_category not 1');
            raw_series_try = person_box_height- person_box_width; % Same as debug_series_data
            N_try = length(raw_series_try);
            lags_try = N_try - 50;
            disp(['DEBUG: TRY block: series length = ', num2str(N_try), ', lags (-50) = ', num2str(lags_try)]);

            if N_try >=2 
                if lags_try > 0 && lags_try < N_try
                    person_box_ACF = autocorr(raw_series_try, 'NumLags', lags_try);
                    disp('DEBUG: autocorr in TRY succeeded. Class of person_box_ACF:'); disp(class(person_box_ACF)); disp('Size:'); disp(size(person_box_ACF));
                else 
                    disp('DEBUG: TRY block: Invalid lags calculated (-50), forcing to CATCH for N-1 lags.');
                    error('Calculated lags for -50 are invalid, using N-1 via catch block.');
                end
            else 
                 disp('DEBUG: TRY block: Series too short for autocorr (N<2), forcing to CATCH.');
                 error('Series too short in TRY (N<2), using N-1 via catch block.');
            end
        catch ME_autocorr_try 
            disp(['DEBUG: CAUGHT error/condition in TRY block autocorr. Message: ', ME_autocorr_try.message]);
            disp('DEBUG: Trying autocorr in CATCH block (using N-1 lags if possible)');
            temp_series_catch = person_box_height - person_box_width; 
            N_catch = length(temp_series_catch);
            disp(['DEBUG: CATCH block: series length = ', num2str(N_catch)]);

            if N_catch >= 2 
                lags_to_use_catch = N_catch - 1;
                disp(['DEBUG: CATCH block: lags_to_use = ', num2str(lags_to_use_catch)]);
                person_box_ACF = autocorr(temp_series_catch, 'NumLags', lags_to_use_catch);
                disp('DEBUG: autocorr in CATCH succeeded. Class of person_box_ACF:'); disp(class(person_box_ACF)); disp('Size:'); disp(size(person_box_ACF));
            else
                disp('DEBUG: CATCH block: Series too short for autocorr (N<2).');
                % person_box_ACF is already [], or will remain [] if try was skipped by error above
                disp('DEBUG: CATCH block: person_box_ACF set/kept as []. Class:'); disp(class(person_box_ACF)); disp('Size:'); disp(size(person_box_ACF));
            end
        end
        
        % Defensive checks before findpeaks
        if ~isnumeric(person_box_ACF) || (~isvector(person_box_ACF) && ~isempty(person_box_ACF))
            disp('Warning: person_box_ACF is not a numeric vector or is not empty and not a vector. Setting to empty for findpeaks.');
            person_box_ACF = [];
        end
        % Ensure it's a column vector if not empty and is a vector
        if ~isempty(person_box_ACF) && isvector(person_box_ACF) && ~iscolumn(person_box_ACF) 
            person_box_ACF = person_box_ACF(:);
        end
        min_peak_dist = floor(length(person_box_ACF) / 2) - 1;
        if min_peak_dist < 1
            min_peak_dist = 1;
        end
        [~,peak_locs] = findpeaks(person_box_ACF,'MinPeakDistance',min_peak_dist);
        
        if isempty(peak_locs)
            action_period = NaN; % No peaks found, so period is undefined or use a default
            disp('DEBUG: No peaks found by findpeaks. action_period set to NaN.');
        else
            peak_locs = peak_locs(:)'; % Ensure peak_locs is a row vector for [0 peak_locs]
            action_period = mean(diff([0 peak_locs]));
            disp(['DEBUG: Peaks found. action_period calculated as: ', num2str(action_period)]);
        end
    end
    
    if strcmp(cls,'lateral lunge')
        action_period = action_period*2; %since we take right and left lunges to be one period
    end

    
    % get the mesh of the reference frame
    standing_frame_vts = verts(:,:, standing_frame);
    % get the LCS for the reference fram (via eigen analysis)
    [localX, localY, localZ] = get_local_coordinate_system(standing_frame_vts,[torso_vts_idx; leg1_idx; leg2_idx], ...
        front_torso_idx,back_torso_idx, left_toes_idx, 0);
    
    left_ankle_original_standing = get_original_pt_location(joints(1,:,standing_frame), person_box_time(:, standing_frame), cams_time(:,standing_frame), im_shape);
    right_ankle_original_standing = get_original_pt_location(joints(6,:,standing_frame), person_box_time(:, standing_frame), cams_time(:,standing_frame), im_shape);
    
    %test by aligning the standing person's mesh and getting where the feet
    %are in the GCS
    standing_frame_vts_adjusted = (standing_frame_vts - mean(standing_frame_vts,1))*[localX, localY, localZ];
    left_ankle_fixed  = mean(standing_frame_vts_adjusted(left_foot_idx,:)) - mean(standing_frame_vts_adjusted([left_foot_idx; right_foot_idx],:));
    right_ankle_fixed  = mean(standing_frame_vts_adjusted(right_foot_idx,:)) - mean(standing_frame_vts_adjusted([left_foot_idx; right_foot_idx],:));
    
    for i = 1 : num_frames %loop over all frames
        verts(:,:,i) = (verts(:,:,i)-mean(verts(:,:,i),1))*[localX, localY, localZ]; %rotate/align the frame
        
        if strcmp(cls, 'lateral lunge') %for lateral lunges, make sure the person does not bend forward in the alignment process
            [U,~,~] = svd(verts(:,:,i)','econ');
            [~,which_col] = max(abs(U(1,:)));
            U = U * sign(U(1,which_col));
            theta = max(atand(U(3,which_col)/ U(1,which_col)),0);
            verts(:,:,i) = verts(:,:,i)*[cosd(theta),0,sind(theta);0,1,0;-sind(theta),0,cosd(theta)]';
            left_foot_check = norm(left_ankle_original_standing - get_original_pt_location(joints(1,:,i), person_box_time(:, i), cams_time(:,i), im_shape));
            right_foot_check = norm(right_ankle_original_standing - get_original_pt_location(joints(6,:,i), person_box_time(:, i), cams_time(:,i), im_shape));
            if left_foot_check >= right_foot_check                    
                verts(:,:,i) = verts(:,:,i) - mean(verts(left_foot_idx,:,i)) + left_ankle_fixed;
            else
                verts(:,:,i) = verts(:,:,i) - mean(verts(right_foot_idx,:,i)) + right_ankle_fixed;
            end
        end        
        if strcmp(cls, 'stiff-leg deadlift') % for stiff-leg deadlift, make sure the legs stay up-right
            [~, ~, localZcurr] = get_local_coordinate_system(verts(:,:,i),[leg1_idx; leg2_idx], ...
                front_torso_idx,back_torso_idx, left_toes_idx, 0);
            theta = acosd(abs(localZcurr(3))/norm([localZcurr(1) localZcurr(3)]))*0.75;
            verts(:,:,i) = verts(:,:,i)*[cosd(theta),0,sind(theta);0,1,0;-sind(theta),0,cosd(theta)]';
            verts(:,:,i) = verts(:,:,i) - mean(verts([left_foot_idx; right_foot_idx],:,i));
        end
    end
    
    period_start_frames = -action_period:action_period: num_frames+action_period;
    [frameshift, frameshift_loc]  = min(abs(standing_frame- period_start_frames));
    period_start_frames = period_start_frames + frameshift*sign(standing_frame-period_start_frames(frameshift_loc));
    period_start_frames(period_start_frames < 0) = [];
    period_start_frames(period_start_frames > num_frames) = [];
    

%% ========================================================================
%% sit up
%% ========================================================================
elseif cls_category == 8
    verts=all_verts_time;
    joints = all_joints_time;
    
    
    % Get the reference frame as the widest frame (person lying down)
    person_box_width = person_box_time(4,:)-person_box_time(2,:);
    [~, lying_frame] = max(person_box_width);
    
    lying_frame_vts = verts(:,:, lying_frame);
    
    % Get the LCS
    [localX, localY, localZ] = get_local_coordinate_system(lying_frame_vts,[torso_vts_idx;right_foot_idx; left_foot_idx], ...
        front_torso_idx,back_torso_idx, left_toes_idx, 0);
    
    
    % Transform all frame (note now the local X axis points to global Z)
    for i = 1 : num_frames
        verts(:,:,i) = (verts(:,:,i)-mean(verts(:,:,i),1))*[ -localZ , localY ,  localX];
        verts(:,:,i) = verts(:,:,i) - mean(verts(hip_idx,:,i));
    end
    
    % Robust autocorr for person_box_width (cls_category == 8)
    series_cat8 = person_box_width;
    N_cat8 = length(series_cat8);
    person_box_ACF_cat8 = []; % Initialize
    disp(['DEBUG: cls_category=8: series length = ', num2str(N_cat8)]);
    if N_cat8 >= 2 % Must have at least 2 points for any ACF
        lags_desired_cat8 = N_cat8 - 50;
        disp(['DEBUG: cls_category=8: desired lags (-50) = ', num2str(lags_desired_cat8)]);
        if lags_desired_cat8 > 0 && lags_desired_cat8 < N_cat8
            person_box_ACF_cat8 = autocorr(series_cat8, 'NumLags', lags_desired_cat8);
            disp('DEBUG: cls_category=8: autocorr with -50 lags SUCCEEDED.');
        else
            lags_fallback_cat8 = N_cat8 - 1;
            disp(['DEBUG: cls_category=8: -50 lags invalid, trying N-1 lags = ', num2str(lags_fallback_cat8)]);
            if lags_fallback_cat8 > 0 % Ensure N-1 is at least 1
                person_box_ACF_cat8 = autocorr(series_cat8, 'NumLags', lags_fallback_cat8);
                disp('DEBUG: cls_category=8: autocorr with N-1 lags SUCCEEDED.');
            else
                disp('DEBUG: cls_category=8: N-1 lags also invalid (<=0). ACF set to [].');
            end
        end
    else
        disp('DEBUG: cls_category=8: Series too short for ACF. ACF set to [].');
    end
    if ~isempty(person_box_ACF_cat8)
        disp(['DEBUG: cls_category=8: Final ACF class: ', class(person_box_ACF_cat8), ', size: [', num2str(size(person_box_ACF_cat8,1)), ' ', num2str(size(person_box_ACF_cat8,2)), ']']);
    end
    person_box_ACF = person_box_ACF_cat8; % Assign to the variable used by findpeaks

    min_peak_dist = floor(length(person_box_ACF) / 2) - 1;
    if min_peak_dist < 1
        min_peak_dist = 1;
    end
    [~,peak_locs] = findpeaks(person_box_ACF,'MinPeakDistance',min_peak_dist);
    action_period = mean(diff([0 peak_locs]));
    period_start_frames = -action_period:action_period: num_frames+action_period;
    [frameshift, frameshift_loc]  = min(abs(lying_frame - period_start_frames));
    period_start_frames = period_start_frames + frameshift*sign(lying_frame-period_start_frames(frameshift_loc));
    period_start_frames(period_start_frames < 0) = [];
    period_start_frames(period_start_frames > num_frames) = [];
    
    

else %unidentified category
    verts = all_verts_time;
    joints = all_joints_time;
end