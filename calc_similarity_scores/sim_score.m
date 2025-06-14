v1 = []; %real
v2 = []; %sim


% Load into a temporary struct
tmp = load('Act10_DL.mat');          
fn     = fieldnames(tmp);            % field name(s) inside the .mat
v1 = tmp.(fn{1});                % copy the contents
clear tmp; 




tmp = load('sit_up.mat');          % tmp is a struct whose single field is the variable
fn     = fieldnames(tmp);            % field name(s) inside the .mat
v2 = tmp.(fn{1});                % copy the contents
clear tmp; 

cosSim = dot(v1, v2) / (norm(v1) * norm(v2));

% optional: display with a label
fprintf('Cosine similarity = %.6f\n', cosSim);

