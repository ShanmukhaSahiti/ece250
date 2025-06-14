%%% Parameters
quantiles = [0.5 0.7];                    % which quantiles to analyse
edges     = [1 10 20 30 40 100];          % bin edges  (Hz)

% Pre-allocate: rows = 3 spectrograms, cols = 5 bins
hist05 = zeros(3,5);      % for the 0.5 quantile
hist07 = zeros(3,5);      % for the 0.7 quantile

%%% Main loop
for k = 1:3
    % ---- Fetch the kth spectrogram and its frequency axis ----
    data = total_spectrogram(:,:,k);
    %if isstruct(data)
    %    S = abs(data.S);          % |magnitude|  (or data.P if already power)
    %    f = data.F(:);            % make sure it is a column vector
    %else                          % plain matrix case
    S = abs(data);            % user must have freqAxis in workspace
        %f = freqAxis(:);
    f = transpose(1:100);
    %end

    % ---- Power per frequency bin ----
    P = S.^2;                     % use power for better energy representation
    cumP = cumsum(P,1);           % cumulative power over frequency (row) axis
    totP = cumP(end,:);           % total power for every time slice

    % ---- For each requested quantile ----
    for q = quantiles
        % energy level that reaches the quantile in every column
        qLevel = q * totP;                                % 1 × Ntime
        % Find, for every column, the first frequency index whose
        % cumulative power meets/exceeds that level
        idx = arrayfun(@(c) find(cumP(:,c) >= qLevel(c),1,'first'), ...
                       1:size(S,2));                      % 1 × Ntime
        qFreq = f(idx);                                   % quantile frequencies (Hz)

        % ---- Histogram over the required frequency bands ----
        counts = histcounts(qFreq,edges);                 % 1 × 5

        % ---- Store ----
        if abs(q-0.5)<eps
            hist05(k,:) = counts;
        else
            hist07(k,:) = counts;
        end
    end
end


histAll = [hist05 , hist07];
histFlat = [histAll(1,:),histAll(2,:),histAll(3,:)];
save('Act9_Subj10.mat', 'histFlat');
%%% Display or return the results
disp('Histogram (rows = spectrograms, cols = bins)');
fprintf('\n0.5-quantile frequencies:\n');  disp(hist05);
fprintf('\n0.7-quantile frequencies:\n');  disp(hist07);
