% process_csi.m
% When called with no inputs, it will iterate over all your files and
% run the pipeline+plots on each one. You can also call process_csi_pipeline
% directly if you just want one file.

function process_csi(file_path, num_samples)
if nargin==0
    % ========== CONFIGURAÇÕES ==========
    clc; clear;
    num_samples = 100;
    base_path   = 'C:\Users\diana\OneDrive\Uni- Di pessoal\MESTRADO\Tese\Dados_csi\data\';

    positions = struct( ...
        'empty', {{...
            'user00_positionz00_esp01_noite.csv', ...
            'user00_positionz00_esp02_noite.csv', ...
            'user00_positionz00_esp03_noite.csv', ...
            'user00_positionz00_esp04_noite.csv'}}, ...
        'c06',   {{...
            'user01_positionc06_esp01.csv', ...
            'user01_positionc06_esp02.csv', ...
            'user01_positionc06_esp03.csv', ...
            'user01_positionc06_esp04.csv'}}, ...
        'a09',   {{...
            'user01_positiona10_esp01.csv', ...
            'user01_positiona10_esp02.csv', ...
            'user01_positiona10_esp03.csv', ...
            'user01_positiona10_esp04.csv'}} ...
    );

    % Loop over each field in positions
    flds = fieldnames(positions);
    for fi = 1:numel(flds)
        pos_name = flds{fi};
        files    = positions.(pos_name);
        for k = 1:numel(files)
            fn = files{k};
            fullfp = fullfile(base_path, fn);
            if ~isfile(fullfp)
                warning('File not found: %s', fullfp);
                continue;
            end
            fprintf('\n=== Processing [%s] %s ===\n', pos_name, fn);
            process_csi_pipeline(fullfp, num_samples);
            pause(0.5);  % let figures render
        end
    end

    return
end

% If we do have inputs, just run one file through the pipeline:
process_csi_pipeline(file_path, num_samples);
end


%%---------------------------------------------------------------------------%%
function process_csi_pipeline(file_path, num_samples)
    % Reads CSI from CSV, processes it, and plots each step
    [magnitudes, RSSI] = read_and_parse_csi(file_path, num_samples);

    % 1) RSSI over time
    figure; plot(1:length(RSSI), RSSI, '-o','LineWidth',1.5);
    title('RSSI over Samples');
    xlabel('Sample Index'); ylabel('RSSI');

    % 2) Raw magnitudes (first frame)
    figure; bar(magnitudes(1,:));
    title('Subcarrier Magnitudes (Frame 1)');
    xlabel('Subcarrier Index'); ylabel('Magnitude');

    % 3) Normalization
    baseline_mean    = mean(magnitudes,1);
    baseline_abs_max = max(abs(magnitudes - baseline_mean), [],1);
    magn_norm = (magnitudes - baseline_mean) ./ baseline_abs_max;
    figure; bar(magn_norm(1,:));
    title('Normalized Magnitudes (Frame 1)');
    xlabel('Subcarrier Index'); ylabel('Normalized');

    % 4) Smoothing
    win = 21;  % moving average window (odd)
    magn_smooth = movmean(magn_norm, win, 1);
    [nsamp, nsub] = size(magn_smooth);
    sel = [1, round(nsub/2), nsub];      % always valid indices
    figure; plot(1:nsamp, magn_smooth(:, sel), 'LineWidth',1.5);
    legend(arrayfun(@(x) sprintf('SC %d', x), sel, 'UniformOutput',false));
    title('Smoothed Signals (Selected Subcarriers)');
    xlabel('Sample Index'); ylabel('Smoothed Magnitude');

    % 5) Segmentation & plot
    overlap_step = 5;
    [segments, t_segments] = segment_signal(magn_smooth, win, overlap_step);
    figure; hold on;
    for kk = 1:numel(segments)
        plot(t_segments{kk}, segments{kk}(:,1));
    end
    hold off;
    title('Overlapping Segments (Subcarrier 1)');
    xlabel('Sample Index'); ylabel('Magnitude');
end


%%---------------------------------------------------------------------------%%
function [magnitudes, RSSI] = read_and_parse_csi(file_path, num_samples)
    data = readtable(file_path, 'TextType','string');
    RSSI = data{1:num_samples, 5};
    csi_raw = data{1:num_samples, 27};  % Var27 holds CSI strings
    valid_magn = [];

    for i = 1:num_samples
        str = extractBetween(csi_raw(i), '[', ']');
        if isempty(str), continue; end
        nums = str2double(regexp(str{1}, '[-]?\d+', 'match'));
        if numel(nums)==128
            cvec = complex(nums(1:2:end), nums(2:2:end));
            cvec = fftshift(cvec);
            active = cvec(7:58);    % 52 active subcarriers
            active(27) = [];        % remove central tone
            valid_magn = [valid_magn; abs(active).'];
        end
    end

    magnitudes = valid_magn;
end


%%---------------------------------------------------------------------------%%
function [segments, t_segments] = segment_signal(data, window_size, step)
    N = size(data,1);
    segments = {};
    t_segments = {};
    idx = 1;
    while idx + window_size - 1 <= N
        segments{end+1}   = data(idx:idx+window_size-1, :);
        t_segments{end+1} = idx:idx+window_size-1;
        idx = idx + step;
    end
end
