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
% 5. Lateral lunge, 9. Sit-up, 10. Stiff-leg deadlift, 11. Swing bench, 12. Lifting
if ~ismember(activity_id, [5,9,10,11,12])
    error('Activity ID %d is not recognized or supported. Please choose from [5, 9, 10, 11, 12].', activity_id)
end

%% Adding paths and pre-loading some data files
addpath('Functions'); %some functions used in the script

% Get available videos for the chosen activity
[cls, cls_category, available_videos] = get_action_name(activity_id);

addpath('MeshInfoMatFiles')
load('left_arm_idx.mat');
load('right_arm_idx.mat');
load('noarm_idx.mat'); 
% Removed debug lines for noarm_idx here
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


% Loop over videos (hardcoded, will cause error if available_videos is shorter)
% This is the state that processed the first video and then failed.
for vid_id = 1:length(available_videos)
    
    % Simplified clearvars, preserving essential loop variables and overall parameters
    % This version assumes cls, cls_category, available_videos are from the outer scope.
    % Adding noarm_head_idx etc. to preserve them if loaded by noarm_idx.mat
    clearvars -except show_meshes activity_id cls cls_category available_videos vid_id ...
                      num_link Tx_pos_all Rx_pos_all name_link beamwidth_body_default ...
                      fc lambda left_arm_idx right_arm_idx set_noarm noarm_idx ...
                      noarm_head_idx noarm_torso_idx noarm_left_leg_idx noarm_right_leg_idx
    
    current_video_id_str = available_videos{vid_id}; % This line will cause an error if vid_id is out of bounds
    fprintf('Class: %s\nVideo: %s\n',cls, current_video_id_str);
    
    % load frames and data
    vid = ['v-' current_video_id_str]; %video name
    folder_frame = fullfile('video_frames',cls,vid);
    frame_all = dir(folder_frame);
    frame_one = imread(fullfile(folder_frame,frame_all(3).name));
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
        
        if strcmp(cls, 'swing bench')
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
        
        % For swing bench: name_parts{1} is like '669-60105'. We need to ensure the box filename matches.
        % The mask files are named: 669-60105_cropped_im_mask.mat
        if strcmp(cls, 'swing bench') || strcmp(cls, 'lifting')
            % The mesh filenames are like '669-60105_cropped_im_mesh.mat'
            % or '2502-2_70340_cropped_im_mesh.mat'
            % The mask files are simpler: '669-60105_mask.mat' or '2502-2_70340_mask.mat'
            
            % We get the base name from the mesh file: e.g., '669-60105_cropped_im_mesh'
            [~, mesh_name_base, ~] = fileparts(mesh_all(iter_mesh).name);
            % It should be '669-60105' for swing bench or '2502-2_70340' for lifting
            
            % The python script creates mesh file names like '..._cropped_im_mesh.mat'
            % and mask file names like '..._mask.mat'.
            % So we need to remove '_cropped_im_mesh' and add '_mask.mat'
            
            box_filename_base = strrep(mesh_name_base, '_cropped_im_mesh', '');
            box_filename_construct = [box_filename_base, '_mask.mat'];
        else
            % Original logic for box filename
            box_filename_construct = [name_parts{1},'_',name_parts{2},'_',name_parts{3}, '_mask.mat'];
        end
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
        %     fprintf('%f\n',box_c);
        person_mask_time{1,iter_mesh} = person_mask;
    end
    
    t = t - t(1); % the time variable
    
    % Adjusting x,y,z of the meshes as they are swapped in the mesh
    % extraction codes
    all_verts_time = [all_verts_time(:,1,:) , all_verts_time(:,3,:), -all_verts_time(:,2,:)];
    all_joints_time = [all_joints_time(:,1,:) , all_joints_time(:,3,:), -all_joints_time(:,2,:)];
    
    %% 3D Mesh Alignment via Eigen Analysis (Sec 3.2 of the paper)
    all_verts_time_unaligned = all_verts_time;
    [all_verts_time, all_joints_time, period_start_frames] = ...
        mesh_alignment_algorithm(all_verts_time, all_joints_time, person_box_time,...
        person_mask_time, cams_time, im_shape,...
        cls, cls_category, current_video_id_str); % Use cls and current_video_id_str
    
    %the times at which periods of the activity start
    period_start_times = interp1(1:num_mesh,t, period_start_frames); % t(round(period_start_frames));
    
    if isnan(period_start_frames) % non-periodic activities, take the entire video as one period
        period_start_frames = 1;
    end
        
    
    % interpolating mesh to uniform time samples and smoothing them over
    % time
    ts = 0.005;
    tq = 0:ts:t(end);
    [vv, ~,tt] = size(all_verts_time);
    all_verts_time2 = reshape(all_verts_time,[vv*3 tt]);
    all_verts_time2 = interp1(t,all_verts_time2',tq,'spline');
    all_verts_time = reshape(all_verts_time2',[vv 3 length(tq)]);
    all_verts_time = smoothdata(all_verts_time,3,'movmean',80);
    
    
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
            xlabel('x');
            ylabel('y');
            zlabel('z');
            axis([-3,3,-3,3,-.5,2.5]);
            view([45,20])
            title('Unaligned (raw, extracted) Mesh')
            
            [~,j] = min(abs(tq-t(i)));
            subplot(1,3,3)            
            scatter3(all_verts_time(:,1,j(1)),all_verts_time(:,2,j(1)),all_verts_time(:,3,j(1)),'.');
            xlabel('x');
            ylabel('y');
            zlabel('z');
            axis([-3,3,-3,3,-.5,2.5]);
            title('Aligned Mesh')
            view([45,20])
            pause(0.05)
        end
    end
    
    
    %% Generating wifi signals
    fprintf('Simulating the WiFi signal\n');
    % Which parts to include in the simulation? Use only the mostly moving
    % ones
    if ismember(activity_id,[9,10,12]) 
        scaling_body = [1,1,0,0];
    elseif ismember(activity_id,[5]) 
        scaling_body = [1,1,1,1];
    elseif ismember(activity_id, [11])
        scaling_body = [1,1,1,1]; % Assuming swing bench uses all parts
    end
    
    % preallocate variables to store spectrograms
    sp_all = cell(num_link,1);
    RS_all = cell(num_link,1);
    
    % min number of mesh points to accept
    min_verts = 50;
    
    % some body parts have very sparse number of mesh points and need to be
    % interpolated (the head to be more specific)
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
            % mesh at current time
            verts_curr = all_verts_time(:,:,i);
            
            % no arm body
            verts_curr = verts_curr(set_noarm,:);
            
            % only visible nodes by Tx and Rx will contribute to reflected
            % wifi signal (see XModal-id)
            idx_visible = intersect(HPR(verts_curr,Tx_pos,2.25),...
                HPR(verts_curr,Rx_pos,2.25)) ;
            
            verts = verts_curr(idx_visible,:);
            scale_body_surface = ones(size(verts,1),1);
            beamwidth = beamwidth_body_default*ones(size(verts,1),1);
            
            if do_parts == 1 % If interpolation is required
                verts = [];
                scale_body_surface = [];
                beamwidth = [];
                % head, torso...
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
                        % interpolate to get denser mesh
                        min_x = min(verts_part_visible(:,1));
                        max_x = max(verts_part_visible(:,1));
                        min_y = min(verts_part_visible(:,2));
                        max_y = max(verts_part_visible(:,2));
                        
                        res = res_interp(iter_part);
                        [xq,yq] = meshgrid(min_x:res:max_x, min_y:res:max_y);
                        zq = griddata(verts_part_visible(:,1),verts_part_visible(:,2),verts_part_visible(:,3),xq,yq);
                        
                        % griddata only inteporlates data from within the convex hull of
                        % existing data, invalid queries return nan
                        idx_valid = find(~isnan(zq(:)));
                        verts_part_visible = [verts_part_visible;[xq(idx_valid),yq(idx_valid),zq(idx_valid)]];
                    end
                    
                    verts = [verts;verts_part_visible];
                    scale_body_surface = [scale_body_surface;...
                        scaling_body(iter_part)*ones(size(verts_part_visible,1),1)];
                    beamwidth = [beamwidth;...
                        beamwidth_body_default*ones(size(verts_part_visible,1),1)];
                end
            end
            
            
            %Calculate incident and reflected rays directions on each mesh
            %point
            normals = pcnormals(pointCloud(verts),12);
            point_Tx_vector = Tx_pos - verts;
            % normals_to_flip = (sign(normals(:,iter_link))~=sign(point_Tx_vector(iter_link))); % Potential error if point_Tx_vector has varying signs
            % A more robust way if normals can be multi-column and point_Tx_vector too:
            % Compare element-wise for the iter_link column, or ensure iter_link is a valid index for point_Tx_vector
            % Assuming point_Tx_vector is N_verts x 3 and normals is N_verts x 3
            % And iter_link is 1, 2, or 3 for x, y, z component comparison.
            if size(point_Tx_vector,2) >= iter_link && size(normals,2) >= iter_link
                normals_to_flip = (sign(normals(:,iter_link)) ~= sign(point_Tx_vector(:,iter_link)));
                normals(normals_to_flip,:) = -normals(normals_to_flip,:);
            else
                % Handle cases where iter_link might be out of bounds for columns, though unlikely here
                % Or if point_Tx_vector was scalar for the iter_link component, original code was fine.
                % For safety, if this occurs, don't flip, or error. Here, just proceed.
            end

            point_rx_vector = Rx_pos - verts;
            point_rx_vector = point_rx_vector ./ sqrt(sum(point_rx_vector.^2,2));
            incident_vector = -point_Tx_vector ./ sqrt(sum(point_Tx_vector.^2,2));
            ref_vector = incident_vector - 2*(diag(incident_vector*normals')).*normals;
            
            angle = acosd(diag(point_rx_vector*ref_vector'));
            scale_ref_beam = exp(-angle.^2/2./beamwidth);
            
            %distances from mesh points to Tx and Rx
            dist = sqrt((Tx_pos(1) - verts(:,1)).^2 + (Tx_pos(2) - verts(:,2)).^2 + (Tx_pos(3) - verts(:,3)).^2) + ...
                sqrt((Rx_pos(1) - verts(:,1)).^2 + (Rx_pos(2) - verts(:,2)).^2 + (Rx_pos(3) - verts(:,3)).^2);
            
            % The BORN APPROXIMATION to calculate the received signal
            RS(i) = sum(scale_body_surface.*scale_ref_beam.*exp(1j*2*pi*dist/lambda)./ 4./pi./dist) + exp(1j*2*pi*dist_Tx_Rx/lambda)/4/pi/dist_Tx_Rx ;
        end
        
        RS_all{iter_link} = RS;
        
        disp('Generating the spectrogram ...')
        %Spectrogram generation
        window= floor(0.4/ts);
        abs_RS = abs(RS).^2;
        abs_RS = remove_low_freq(abs_RS, 0.15/ts);
        freq = [1:1:100];
        [sp,F,T] = spectrogram(abs_RS,window,window - 1,freq,1/ts);
        sp = abs(sp);
        sp_all{iter_link} = sp;
    end
    
    %% plotting the spectrograms
    
    % denoise the spectrogram ?
    do_denoise = 0;
    noise_floors = [0.05, 0.05, 0.02];
    
    close all;
    figure;
    for iter_link = set_link
        video_spectrogram=abs(sp_all{iter_link});
        
        % denoise
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