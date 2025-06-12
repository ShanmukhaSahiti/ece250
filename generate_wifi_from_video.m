function generate_wifi_from_video(activity_id_input)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This code implements the video-to-wifi simulation pipeline of the     %%
%% paper, "Teaching RF to sense without RF training measurements",       %%
%% published in IMWUT20 (vol. 4, issue 4), with application to the three %%
%% sample GYM activities (lateral lunge, sit up, and deadlift).          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c): H. Cai, B. Korany, and C. Karanam (UCSB, 2020)         %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ensure we're using the function argument
activity_id = activity_id_input;
fprintf('generate_wifi_from_video: Received activity_id = %d\n', activity_id);

% clear functions; % Generally not needed/desired inside a function called this way
% rehash toolboxcache; % Generally not needed/desired

% clear; % CANNOT use clear as it would wipe the input argument activity_id_input
clc; % Safe
close all; % Safe

% Validate activity_id
% 11. Swing bench, 12. Lifting, 13. Diving-Side, 14. Golf-Swing-Side, 15. Kicking-Front, 16. run-side, 17. skateboarding-front, 18. walk-front
if ~ismember(activity_id, [11, 12, 13, 14, 15, 16, 17, 18])
    error('Activity ID %d is not recognized or supported. Please choose from [11, 12, 13, 14, 15, 16, 17, 18].', activity_id)
end

%% Adding paths and pre-loading some data files
addpath('Functions'); %some functions used in the script

% Get available videos for the chosen activity
[cls, cls_category, available_videos] = get_action_name(activity_id);

addpath('MeshInfoMatFiles')
load('left_arm_idx.mat');
load('right_arm_idx.mat');
load('noarm_idx.mat'); 
set_noarm = setdiff(1:6890,[left_arm_idx;right_arm_idx]); %indeces of all body points except arms

%% pre-setting some simulation parameters
fc = 5.18e9; %Frequency of the wireless signal
lambda = 3e8/fc; %wavelength of the wireless signal
Tx_pos_all = [2,   2,  0.76; 2,   2,  0.76; -0.20,   0.25,  2.75]; %3D locations of the Tx's, see Fig.3 in the paper
Rx_pos_all = [2,  -2,  0.76;-2,   2,  0.76; -0.20,  -0.25,  2.75]; %3D locations of the Rx's, see Fig.3 in the paper
num_link = size(Tx_pos_all,1); % number of links
name_link = {'x','y','z'}; %link names
beamwidth_body_default = 40;%beamwidth of the specular reflection of different body parts, refer to paper


%% Visualization parameters (visualizing the moving mesh)
show_meshes = 0; %Binary to show the mesh or not


% Loop over videos
for vid_id = 1:length(available_videos)
    
    % Simplified clearvars, preserving essential loop variables and overall parameters
    clearvars -except show_meshes activity_id cls cls_category available_videos vid_id ...
                      num_link Tx_pos_all Rx_pos_all name_link beamwidth_body_default ...
                      fc lambda left_arm_idx right_arm_idx set_noarm noarm_idx ...
                      noarm_head_idx noarm_torso_idx noarm_left_leg_idx noarm_right_leg_idx
    
    current_video_id_str = available_videos{vid_id};
    fprintf('Class: %s\nVideo: %s\n',cls, current_video_id_str);
    
    % load frames and data
    if startsWith(current_video_id_str, 'v-') || startsWith(current_video_id_str, 'v')
        vid = current_video_id_str;
    else
        vid = ['v-' current_video_id_str];
    end

    folder_frame = fullfile('video_frames',cls,vid);
    frame_all = dir(fullfile(folder_frame, '*.jpg')); % Get only .jpg files
    if isempty(frame_all)
        error('The directory "%s" does not contain any image files. Please ensure it is populated correctly.', folder_frame);
    end
    frame_one = imread(fullfile(folder_frame,frame_all(1).name)); % Read the first .jpg file
    [h_frame,w_frame,~] = size(frame_one); % just to get image size here and nothing else
    
    im_shape = [h_frame,w_frame];
    folder_mesh = fullfile('video_meshes',cls, [vid,'_mat_mesh']);
    folder_box = fullfile('video_meshes',cls, [vid,'_mat_mask']);
    folder_cropped_im =  fullfile('video_meshes',cls, [vid,'_cropped_im']);
    
    mesh_all = dir(fullfile(folder_mesh, '*.mat'));
    if isempty(mesh_all)
        fprintf('No mesh files found for %s, video %s. Skipping.\n', cls, vid);
        continue; % Skip to next video iteration
    end
    [~,srt_idx] = natsortfiles({mesh_all.name}); % indices of natural order
    mesh_all = mesh_all(srt_idx); % sort structure using indices
    num_mesh = length(mesh_all); %number of video frames
    
    cropped_im_all = dir(fullfile(folder_cropped_im, '*.jpg'));
    cropped_im_all = cropped_im_all(srt_idx);
    
    % pre-allocation
    all_verts_time = zeros(6890,3,num_mesh); %3d coordinates of all mesh points across time
    all_joints_time = zeros(19,3,num_mesh); %3d coordinates of all joints across time
    person_box_time = zeros(4,num_mesh); %the bounding box of the person in all frames
    person_mask_time = cell(1,num_mesh);
    box_h_time = zeros(1, num_mesh);
    box_c_time = zeros(1, num_mesh);
    cams_time = zeros(3, num_mesh); %the camera angle in all frames
    
    disp('Obtaining mesh and aligning it ...')
    for iter_mesh = 1:num_mesh %looping over the mesh in all frames and saving them in the previously preallocated variables
        load(fullfile(folder_mesh,mesh_all(iter_mesh).name));
        verts = squeeze(verts);
        joints3d = squeeze(joints3d);
        all_verts_time(:,:,iter_mesh) = verts;
        all_joints_time(:,:,iter_mesh) = joints3d;
        cams_time(:,iter_mesh) = cams';
        cropped_images{iter_mesh} = imread(fullfile(folder_cropped_im,cropped_im_all(iter_mesh).name));
        
        name_parts = strsplit(mesh_all(iter_mesh).name,'_');
        
        if strcmp(cls, 'Swing-Bench')
            % Handle filenames like '669-60105_cropped_im_mesh.mat'
            if length(name_parts) >= 1
                id_part = strsplit(name_parts{1}, '-');
                if length(id_part) >= 2
                    t(iter_mesh) = str2double(id_part{2});
                else
                    warning('Could not parse frame number from swing bench mesh filename: %s (part 1 after hyphen)', mesh_all(iter_mesh).name);
                    t(iter_mesh) = iter_mesh; % Fallback to iteration index
                end
            else
                warning('Could not parse frame number from swing bench mesh filename: %s (part 1)', mesh_all(iter_mesh).name);
                t(iter_mesh) = iter_mesh; % Fallback to iteration index
            end
        elseif strcmp(cls, 'lifting')
             % Handle filenames like '2502-2_70352_cropped_im_mesh.mat'
             if length(name_parts) >= 2
                 t(iter_mesh) = str2double(name_parts{2});
             else
                 warning('Could not parse frame number from lifting mesh filename: %s', mesh_all(iter_mesh).name);
                 t(iter_mesh) = iter_mesh; % Fallback
             end
        else
            % Original parsing logic for other activities
            if length(name_parts) >= 3 && length(name_parts{3}) >= 6 % Basic check
                if isnan(str2double(name_parts{3}(end-5)))
                    t(iter_mesh) = str2double(name_parts{3}(end-4:end));
                else
                    t(iter_mesh) = str2double(name_parts{3}(end-5:end));
                end
            else
                warning('Could not parse frame number from mesh filename: %s using original logic', mesh_all(iter_mesh).name);
                t(iter_mesh) = iter_mesh; % Fallback to iteration index
            end
        end
        
        % The python script creates mesh file names like '..._cropped_im_mesh.mat'
        % and mask file names like '..._mask.mat'.
        % So we need to remove '_cropped_im_mesh' and add '_mask.mat'
        [~, mesh_name_base, ~] = fileparts(mesh_all(iter_mesh).name);
        box_filename_base = strrep(mesh_name_base, '_cropped_im_mesh', '');
        box_filename_construct = [box_filename_base, '_mask.mat'];

        load(fullfile(folder_box, box_filename_construct));
        person_box = double(person_box);
        person_box_time(:,iter_mesh) = person_box';
        % distance of box from bottom
        box_h = h_frame-person_box(3);
        % roughly normalize it as human mesh is normalized
        box_h = box_h/500;
        box_h_time(iter_mesh) = box_h;
        % box center, and normalize it
        box_c = (person_box(4)+person_box(2))/2-w_frame/2;
        box_c = box_c/400;
        box_c_time(iter_mesh) = box_c;
        person_mask_time{1,iter_mesh} = person_mask;
    end
    
    t = t - t(1); % the time variable
    
    % Adjusting x,y,z of the meshes
    all_verts_time = [all_verts_time(:,1,:) , all_verts_time(:,3,:), -all_verts_time(:,2,:)];
    all_joints_time = [all_joints_time(:,1,:) , all_joints_time(:,3,:), -all_joints_time(:,2,:)];
    
    %% 3D Mesh Alignment via Eigen Analysis
    all_verts_time_unaligned = all_verts_time;
    [all_verts_time, all_joints_time, period_start_frames] = ...
        mesh_alignment_algorithm(all_verts_time, all_joints_time, person_box_time,...
        person_mask_time, cams_time, im_shape,...
        cls, cls_category, current_video_id_str);
    
    period_start_times = interp1(1:num_mesh,t, period_start_frames);
    
    if any(isnan(period_start_times)) || any(isinf(period_start_times))
        warning('Could not determine a valid period for activity %d, video %s. Skipping interpolation and smoothing.', activity_id, current_video_id_str);
        tq = t; % Use original time vector
        ts = mean(diff(tq)); % Calculate ts from the original time vector
        if isnan(ts) || isinf(ts) || ts <= 0 || floor(0.4/ts) < 1
            ts = 1/30; % Fallback to a default if calculation fails
        end
    else
        % interpolating mesh to uniform time samples and smoothing
        ts = 0.005;
        tq = 0:ts:t(end);
        
        % Ensure time vector is unique before interpolation
        [t_unique, ia, ~] = unique(t);
        if length(t_unique) < length(t)
            warning('Duplicate timestamps detected in video %s. Using unique timestamps for interpolation.', current_video_id_str);
            verts_data_for_interp = all_verts_time(:,:,ia);
            time_for_interp = t_unique;
        else
            verts_data_for_interp = all_verts_time;
            time_for_interp = t;
        end
        
        % Check for enough points for spline interpolation
        if length(time_for_interp) < 4
            warning('Not enough unique timestamps (%d) for spline interpolation in video %s. Skipping interpolation.', length(time_for_interp), current_video_id_str);
            tq = t; % Revert to original time vector
            ts = mean(diff(tq));
            if isnan(ts) || isinf(ts) || ts == 0
                ts = 1/30; % Fallback
            end
        else
            [vv, ~,tt] = size(verts_data_for_interp);
            all_verts_time2 = reshape(verts_data_for_interp,[vv*3 tt]);
            all_verts_time2 = interp1(time_for_interp,all_verts_time2',tq,'spline');
            all_verts_time = reshape(all_verts_time2',[vv 3 length(tq)]);
            all_verts_time = smoothdata(all_verts_time,3,'movmean',80);
        end
    end
        
    
    %% Visualization
    if show_meshes
        disp('Showing the meshes')
        figure()
        set(gcf, 'Position', [100 200 1500 400]);
        for i = 1 : num_mesh
            
            subplot(1,3,1) 
            imshow(cropped_images{i});
			title('Original Video')
            
            subplot(1,3,2)
            scatter3(all_verts_time_unaligned(:,1,i),all_verts_time_unaligned(:,2,i),all_verts_time_unaligned(:,3,i),'.');
            xlabel('x'); ylabel('y'); zlabel('z');
            axis([-3,3,-3,3,-.5,2.5]); view([45,20])
            title('Unaligned (raw, extracted) Mesh')
            
            [~,j] = min(abs(tq-t(i)));
            subplot(1,3,3)            
            scatter3(all_verts_time(:,1,j(1)),all_verts_time(:,2,j(1)),all_verts_time(:,3,j(1)),'.');
            xlabel('x'); ylabel('y'); zlabel('z');
            axis([-3,3,-3,3,-.5,2.5]);
            title('Aligned Mesh')
            view([45,20])
            pause(0.05)
        end
    end
    
    
    %% Generating wifi signals
    fprintf('Simulating the WiFi signal\n');
    % Which parts to include in the simulation?
    if ismember(activity_id, [12]) % lifting
        scaling_body = [1,1,0,0];
    elseif ismember(activity_id, [11, 13, 14, 15, 16, 17, 18]) % swing bench & new activities
        scaling_body = [1,1,1,1];
    end
    
    % preallocate variables
    sp_all = cell(num_link,1);
    RS_all = cell(num_link,1);
    
    min_verts = 50;
    
    do_parts = 1;
    do_interp = [0,0,0,0,0;...
        0,0,0,0,0;...
        1,0,0,0,0];
    res_interp = [0.01,0.02,0.01,0.01,0.01];
    
    set_link = [1,2,3];
    
    for iter_link = set_link
        Tx_pos = Tx_pos_all(iter_link,:);
        Rx_pos = Rx_pos_all(iter_link,:);
        dist_Tx_Rx = norm(Tx_pos - Rx_pos);
        
        disp(['Generating the WiFi Signal at link: ', name_link{iter_link}, ' ...'])
        for i = 1:length(tq)
            verts_curr = all_verts_time(:,:,i);
            verts_curr = verts_curr(set_noarm,:);
            
            idx_visible = intersect(HPR(verts_curr,Tx_pos,2.25),...
                HPR(verts_curr,Rx_pos,2.25)) ;
            
            verts = verts_curr(idx_visible,:);
            scale_body_surface = ones(size(verts,1),1);
            beamwidth = beamwidth_body_default*ones(size(verts,1),1);
            
            if do_parts == 1
                verts = [];
                scale_body_surface = [];
                beamwidth = [];
                idx_included = [];
                for iter_part = 1:length(scaling_body)
                    if iter_part == 1
                        idx_part = intersect(idx_visible, noarm_head_idx); 
                        verts_part_visible = verts_curr(idx_part,:);
                    elseif iter_part == 2
                        idx_part = intersect(idx_visible, noarm_torso_idx);
                        verts_part_visible = verts_curr(idx_part,:);
                    elseif iter_part == 3
                        idx_part = intersect(idx_visible, noarm_left_leg_idx);
                        verts_part_visible = verts_curr(idx_part,:);
                    elseif iter_part == 4
                        idx_part = intersect(idx_visible, noarm_right_leg_idx);
                        verts_part_visible = verts_curr(idx_part,:);
                    else 
                        verts_part_visible = [];
                    end
                    
                    if scaling_body(iter_part) == 0
                        continue;
                    end
                    
                    if do_interp(iter_link,iter_part) == 1
                        if size(verts_part_visible,1) > 3 % Check for enough points for triangulation
                            min_x = min(verts_part_visible(:,1));
                            max_x = max(verts_part_visible(:,1));
                            min_y = min(verts_part_visible(:,2));
                            max_y = max(verts_part_visible(:,2));
                            
                            res = res_interp(iter_part);
                            [xq,yq] = meshgrid(min_x:res:max_x, min_y:res:max_y);
                            zq = griddata(verts_part_visible(:,1),verts_part_visible(:,2),verts_part_visible(:,3),xq,yq);
                            
                            idx_valid = find(~isnan(zq(:)));
                            verts_part_visible = [verts_part_visible;[xq(idx_valid),yq(idx_valid),zq(idx_valid)]];
                        else
                            warning('Skipping interpolation for part %d due to insufficient points (%d).', iter_part, size(verts_part_visible,1));
                        end
                    end
                    
                    verts = [verts;verts_part_visible];
                    scale_body_surface = [scale_body_surface;...
                        scaling_body(iter_part)*ones(size(verts_part_visible,1),1)];
                    beamwidth = [beamwidth;...
                        beamwidth_body_default*ones(size(verts_part_visible,1),1)];
                end
            end
            
            normals = pcnormals(pointCloud(verts),12);
            point_Tx_vector = Tx_pos - verts;
            if size(point_Tx_vector,2) >= iter_link && size(normals,2) >= iter_link
                normals_to_flip = (sign(normals(:,iter_link)) ~= sign(point_Tx_vector(:,iter_link)));
                normals(normals_to_flip,:) = -normals(normals_to_flip,:);
            end

            point_rx_vector = Rx_pos - verts;
            point_rx_vector = point_rx_vector ./ sqrt(sum(point_rx_vector.^2,2));
            incident_vector = -point_Tx_vector ./ sqrt(sum(point_Tx_vector.^2,2));
            ref_vector = incident_vector - 2*(diag(incident_vector*normals')).*normals;
            
            angle = acosd(diag(point_rx_vector*ref_vector'));
            scale_ref_beam = exp(-angle.^2/2./beamwidth);
            
            dist = sqrt(sum((Tx_pos - verts).^2,2)) + sqrt(sum((Rx_pos - verts).^2,2));
            
            RS(i) = sum(scale_body_surface.*scale_ref_beam.*exp(1j*2*pi*dist/lambda)./ 4./pi./dist) + exp(1j*2*pi*dist_Tx_Rx/lambda)/4/pi/dist_Tx_Rx ;
        end
        
        RS_all{iter_link} = RS;
        
        disp('Generating the spectrogram ...')
        %Spectrogram generation
        window= floor(0.4/ts);
        w_low_freq = floor(0.15/ts);
        abs_RS = abs(RS).^2;
        abs_RS = remove_low_freq(abs_RS, w_low_freq);
        freq = [1:1:100];
        [sp,F,T] = spectrogram(abs_RS,window,window - 1,freq,1/ts);
        sp = abs(sp);
        sp_all{iter_link} = sp;
    end
    
    %% plotting the spectrograms
    do_denoise = 0;
    noise_floors = [0.05, 0.05, 0.02];
    
    close all;
    figure;
    for iter_link = set_link
        video_spectrogram=abs(sp_all{iter_link});
        
        if do_denoise == 1
            idx_noise = find(video_spectrogram<noise_floors(iter_link));
            video_spectrogram(idx_noise) = 0;
        end
        
        if iter_link == 1
            subplot(311);
        elseif iter_link == 2
            subplot(312);
        elseif iter_link == 3
            subplot(313);
        end
        
        video_spectrogram_bd = video_spectrogram;
        video_spectrogram_g = imgaussfilt(video_spectrogram_bd,2,'FilterSize',5);
        
        surf(T, F, video_spectrogram_g,'LineStyle','none'); colormap(jet);
        xlabel('time'); ylabel('freq');
        title(sprintf('action: %d (%s), Video: (%s), link: %s',activity_id,cls,current_video_id_str,name_link{iter_link}));
        axis([T(1), T(end), F(1), F(end)]);
        view(2);
    end
    
    %save data
    folder_save = 'simulated_spectrograms';
    folder_save_cls = fullfile(folder_save,cls);
    if ~exist(folder_save_cls,'dir')
        mkdir(folder_save_cls);
    end
    save(fullfile(folder_save_cls,sprintf('vid-%s.mat',current_video_id_str)),...
        'sp_all','T','F','period_start_times');
end