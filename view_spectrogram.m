function view_spectrogram(arg1, arg2)
% VIEW_SPECTROGRAM Loads and displays a pre-generated spectrogram.
%   Can be called in two ways:
%   1. view_spectrogram(activity_id, video_id_str)
%   2. view_spectrogram(full_path_to_mat_file)
%
%   INPUTS:
%   activity_id  = The numeric ID of the activity (e.g., 11, 12, 13, ...)
%   video_id_str = The string identifier for the video (e.g., '001', 'v1', '2502-2')
%   full_path_to_mat_file = A string containing the full path to a .mat file.
%
%   EXAMPLES:
%   >> view_spectrogram(13, '001')
%   >> view_spectrogram('/Users/user/project/simulated_spectrograms/Diving-Side/vid-001.mat')

name_link = {'x','y','z'}; %link names
use_full_path_title = false;

% --- Argument Handling ---
if nargin == 1 && ischar(arg1)
    % Mode 2: Full path provided
    file_path = arg1;
    
    % Parse info from file path for the title
    [path, name, ~] = fileparts(file_path);
    video_id_for_title = strrep(name, 'vid-', '');
    [~, cls_for_title, ~] = fileparts(path);
    use_full_path_title = true;

elseif nargin == 2 && isnumeric(arg1) && (ischar(arg2) || isstring(arg2))
    % Mode 1: Activity ID and Video ID provided
    activity_id = arg1;
    video_id_str = arg2;
    addpath('Functions');
    [cls, ~, ~] = get_action_name(activity_id);
    file_path = fullfile('simulated_spectrograms', cls, sprintf('vid-%s.mat', video_id_str));
    
else
    error(['Invalid input. Use one of the following syntaxes:\n' ...
           'view_spectrogram(activity_id, video_id_str)\n' ...
           'view_spectrogram(full_path_to_mat_file)']);
end


% Check if the file exists
if ~exist(file_path, 'file')
    fprintf('Error: Could not find the spectrogram file at:\n%s\n', file_path);
    fprintf('Please ensure the path is correct or you have run the simulation first.\n');
    return;
end

% Load the data from the file
fprintf('Loading spectrogram data from: %s\n', file_path);
load(file_path, 'sp_all', 'T', 'F');

% Create the plot
figure;
set(gcf, 'Position', [100 200 500 700]); % Set figure size and position

for iter_link = 1:3
    video_spectrogram = abs(sp_all{iter_link});
    
    % Apply Gaussian filter for smoothing, same as in the generation script
    video_spectrogram_g = imgaussfilt(video_spectrogram, 2, 'FilterSize', 5);
    
    % Select the subplot
    subplot(3,1,iter_link);
    
    % Plot the spectrogram surface
    surf(T, F, video_spectrogram_g, 'LineStyle', 'none');
    colormap(jet);
    view(2); % View from top-down
    
    % Add labels and title
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    if use_full_path_title
        title(sprintf('Activity: %s, Video: %s, Link: %s', cls_for_title, video_id_for_title, name_link{iter_link}));
    else
        title(sprintf('Activity: %s (%d), Video: %s, Link: %s', cls, activity_id, video_id_str, name_link{iter_link}));
    end
    axis([T(1), T(end), F(1), F(end)]);
    
end

end 