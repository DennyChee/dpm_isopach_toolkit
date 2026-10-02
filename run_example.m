% Purpose:  STEP-BY-STEP EXAMPLE of the DPM -> ash thickness toolkit. Read it top to
%           bottom; run it all at once (F5) or one section at a time (Ctrl+Enter in
%           each %% section). To use YOUR data, change only the "EDIT HERE" block.
%
%           What it shows:
%             1. what the input looks like (N-by-2: DPM, thickness in mm)
%             2. how to run the fit  -> 4 figures saved to output_directory
%             3. how to read the result (equation, uncertainty, hold-out error)
%             4. how to predict thickness for a few DPM values
%             5. how to turn a whole DPM MAP into a thickness map
%
% Inputs:   example_dpm_thickness.csv (synthetic; created automatically if missing)
% Outputs:  example_run_01_boxplot, _02_fit, _03_bootstrap, _04_plot2 (.png/.pdf)
%           and example_run_05_map (.png/.pdf), all in output_directory
% Date:     2026-10-02

clear; clc; close all;

%% ==========================================================================
%  EDIT HERE - the only lines you need to change for your own data
%  ==========================================================================
toolkit_directory = fileparts(mfilename('fullpath'));         % folder of this script = the toolkit
input_csv_file    = fullfile(toolkit_directory, 'example_dpm_thickness.csv');   % your 2-column CSV
isopach_edges_mm  = [0.1 1 5 10 50 100 200 500];   % your isopach class boundaries, mm, increasing, > 0
output_directory  = fullfile(toolkit_directory, 'example_run_output');          % figures go here
output_basename   = 'example_run';                 % figure file name prefix
% ==========================================================================

%% 0. Setup: make the toolkit functions visible to MATLAB
if isempty(toolkit_directory)
    % mfilename is empty when this code is pasted into the Command Window: assume we are in the toolkit
    toolkit_directory = pwd;
    input_csv_file = fullfile(toolkit_directory, 'example_dpm_thickness.csv');
    output_directory = fullfile(toolkit_directory, 'example_run_output');
end
example_csv_file = fullfile(toolkit_directory, 'example_dpm_thickness.csv');   % the bundled synthetic example
fprintf('\n===== SECTION 0 of 5: setup =====\nAdding the toolkit to the MATLAB path: %s\n', toolkit_directory);
addpath(toolkit_directory);
if ~isfile(input_csv_file)
    if strcmp(input_csv_file, example_csv_file)
        % Only for the example: write the synthetic CSV (same data as make_example_dpm_csv.m)
        fprintf('  %s not found - writing the synthetic example data...\n', input_csv_file);
        write_example_csv(input_csv_file);
    else
        % Never fall back to the example silently when YOUR file is missing
        error('run_example:fileNotFound', 'Input file not found: %s\nCheck input_csv_file in the EDIT HERE block.', input_csv_file);
    end
end
if strcmp(input_csv_file, example_csv_file)
    fprintf('  NOTE: using the SYNTHETIC example data. Set input_csv_file in the EDIT HERE block for your own data.\n');
end

%% 1. The input: one row per pixel, column 1 = DPM (0-1), column 2 = thickness (mm)
fprintf('\n===== SECTION 1 of 5: the input =====\nLoading %s...\n', input_csv_file);
data_matrix = readmatrix(input_csv_file);     % a header row like "dpm,thickness_mm" is skipped
fprintf('  Loaded: shape=%s, dtype=%s\n', mat2str(size(data_matrix)), class(data_matrix));
fprintf('  First rows (dpm, thickness_mm):\n');
disp(data_matrix(1:min(5, size(data_matrix, 1)), :));
% Already have the data in MATLAB? Skip the file and build the matrix directly:
%   data_matrix = [dpm_vector(:), thickness_vector_mm(:)];

%% 2. Run the fit
% One call does everything: box plot, inverse-sigmoid fit, 1000-round out-of-bag
% bootstrap (uncertainty + hold-out test), prediction interval, and 4 figures.
% All options are optional. The most useful ones are shown here with their DEFAULT values,
% EXCEPT MaxPixelsPerClass (default 5000), which is set to 50 only so the example mimics
% real data (~50 pixels per class); the example CSV has 500 per class, so 90% are left out.
% Remove that line for your own data. Defaults live in the DEFAULT SETTINGS block at the
% top of fit_thickness_from_dpm.m.
fprintf('\n===== SECTION 2 of 5: run the fit =====\n');
t0 = tic;
fit_results = fit_thickness_from_dpm(data_matrix, isopach_edges_mm, ...
    'MaxPixelsPerClass', 50, ...            % EXAMPLE ONLY - remove this line for your own data
    'FitTarget', 'class_medians', ...       % 'class_medians' (fit the box-plot medians) or 'pixels'
    'NumBootstrap', 1000, ...               % bootstrap rounds
    'Coverage', 0.95, ...                   % single-pixel prediction interval coverage
    'OutputDirectory', output_directory, ...
    'OutputBasename', output_basename);
fprintf('  Fit finished in %.1fs. Figures are in %s\n', toc(t0), output_directory);
% If the data are too small (a class with < 10 pixels) the function STOPS and tells
% you which classes and which line to edit - merge classes via isopach_edges_mm.

%% 3. Read the result
fprintf('\n===== SECTION 3 of 5: the result =====\nThe fitted function (thickness t in mm):\n');
fprintf('  log10 t = log10(%.2f) + (1/%.3f) * ln( (DPM - %.3f) / (%.3f - DPM) )\n', ...
    fit_results.t50_mm, fit_results.steepness_per_log10_mm, fit_results.floor_dpm, fit_results.ceiling_dpm);
parameter_interval = fit_results.parameter_ci_95;     % 2-by-4: rows = [2.5%; 97.5%], columns = [k, log10 t50, floor, ceiling]
fprintf('  t50       = %6.2f mm   (95%% CI %.2f - %.2f): thickness where DPM is half-way between floor and ceiling\n', ...
    fit_results.t50_mm, 10 ^ parameter_interval(1, 2), 10 ^ parameter_interval(2, 2));
fprintf('  steepness = %6.3f      (95%% CI %.3f - %.3f) per log10 mm\n', ...
    fit_results.steepness_per_log10_mm, parameter_interval(1, 1), parameter_interval(2, 1));
fprintf('  floor     = %6.3f      (95%% CI %.3f - %.3f): DPM background-noise level; below it thickness is not resolvable\n', ...
    fit_results.floor_dpm, parameter_interval(1, 3), parameter_interval(2, 3));
fprintf('  ceiling   = %6.3f      (95%% CI %.3f - %.3f): DPM saturation level\n', ...
    fit_results.ceiling_dpm, parameter_interval(1, 4), parameter_interval(2, 4));
fprintf('  Hold-out error (out-of-bag): RMSE %.3f log10 mm (typical, 1-sigma-like factor x%.2f); %.0f%% of pixels within x2\n', ...
    fit_results.oob.rmse_log10_mm, 10 ^ fit_results.oob.rmse_log10_mm, 100 * fit_results.oob.within_factor_2);
fprintf('  Single-pixel %.0f%% prediction interval: x%.2f to x%.2f around the predicted thickness\n', ...
    100 * fit_results.prediction_interval.coverage, fit_results.prediction_interval.factor);
fprintf('  DPM range used for the fit: %.3f - %.3f (outside it = extrapolation)\n', fit_results.observed_dpm_range);
fprintf('  (scores are for the class-balanced sample - about equal pixels per class - not for a whole map)\n');
fprintf('  The equation is exact between floor and ceiling; DPM at/beyond them gets no thickness (see flags).\n');

%% 4. Predict thickness for a few DPM values
fprintf('\n===== SECTION 4 of 5: predict a few DPM values =====\n');
example_dpm_values = [0.03 0.10 0.25 0.40 0.60 0.80 0.95]';
prediction_table = predict_thickness_from_dpm(fit_results, example_dpm_values);
disp(prediction_table);
% How to read the table:
%   thickness_mm          best estimate (a typical thickness, not the mean)
%   ci_low/high_mm        where the CURVE lies (95%)       -> "on average, this DPM means..."
%   pi_low/high_mm        where ONE PIXEL's thickness lies -> "at this pixel, thickness is..."
%   flag/note             0 ok, 1 extrapolation (DPM outside the fitted range),
%                         2 below floor (thickness not resolvable), 3 above ceiling (saturated),
%                         4 input was NaN, 5 thickness outside the calibrated isopach range
%                         (value kept, e.g. 2000 mm when the thickest isopach is 500 mm,
%                         but NOT supported by data - read it as "at least ~500 mm")

%% 5. Apply to a whole DPM map
% Here a small SYNTHETIC map is made so that the "true" thickness is known. For a real
% DPM GeoTIFF use (Mapping Toolbox):
%   [dpm_map, spatial_reference] = readgeoraster('dpm.tif', 'OutputType', 'double');
%   dpm_map(dpm_map == nodata_value) = NaN;      % mask nodata / water / layover first
%   dpm_map = dpm_map / 100;                      % only if your DPM is in percent
fprintf('\n===== SECTION 5 of 5: apply to a DPM map =====\n');
[dpm_map, true_thickness_map_mm, x_km, y_km] = make_synthetic_dpm_map();
fprintf('  DPM map: shape=%s, dtype=%s\n', mat2str(size(dpm_map)), class(dpm_map));
[~, thickness_maps] = predict_thickness_from_dpm(fit_results, dpm_map);   % two outputs = map mode
% thickness_maps.thickness_mm, .ci_low_mm, .ci_high_mm, .pi_low_mm, .pi_high_mm, .flag
% all have the same size as dpm_map, ready to write back to a GeoTIFF, e.g.
%   geotiffwrite('thickness_mm.tif', single(thickness_maps.thickness_mm), spatial_reference);

% Compare with the known synthetic truth where a thickness was predicted. This is a
% SELF-CONSISTENCY check (the map uses the same forward model as the example data), not a
% validation, and it EXCLUDES pixels flagged 2-4, where DPM carries no thickness information.
predicted_mask = isfinite(thickness_maps.thickness_mm);   % flags 0, 1, 5 have a thickness
map_error_log10_mm = log10(thickness_maps.thickness_mm(predicted_mask)) - log10(true_thickness_map_mm(predicted_mask));
fprintf('  Map pixels with a prediction: %d of %d (%.1f%%); flags: %d below floor, %d above ceiling, %d masked (NaN), %d beyond calibration\n', ...
    sum(predicted_mask(:)), numel(dpm_map), 100 * mean(predicted_mask(:)), ...
    sum(thickness_maps.flag(:) == 2), sum(thickness_maps.flag(:) == 3), sum(thickness_maps.flag(:) == 4), sum(thickness_maps.flag(:) == 5));
fprintf('  Error vs synthetic truth on those pixels: RMSE %.3f log10 mm, %.0f%% within x2\n', ...
    sqrt(mean(map_error_log10_mm .^ 2)), 100 * mean(abs(map_error_log10_mm) <= log10(2)));

% Figure: DPM map, true thickness, predicted thickness, flags
log10_colour_limits = log10([isopach_edges_mm(1), isopach_edges_mm(end)]);
figure_map = figure('Position', [100 100 1300 950]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
imagesc(x_km, y_km, dpm_map, 'AlphaData', ~isnan(dpm_map));
axis image; set(gca, 'YDir', 'normal', 'Color', [0.85 0.85 0.85]);   % grey = masked (NaN), e.g. water
colormap(gca, gray);
colorbar_handle = colorbar; colorbar_handle.Label.String = 'DPM (dimensionless, 0-1)';
clim([0 1]);
xlabel('Easting (km)'); ylabel('Northing (km)');
title({'(a) Input: DPM map (synthetic)', 'grey = masked'});

nexttile;
imagesc(x_km, y_km, log10(true_thickness_map_mm));
axis image; set(gca, 'YDir', 'normal');
colormap(gca, parula);
colorbar_handle = colorbar; colorbar_handle.Label.String = 'log_{10} ash thickness (log_{10} mm)';
clim(log10_colour_limits);
xlabel('Easting (km)'); ylabel('Northing (km)');
title({'(b) True thickness (synthetic)', ' '});

nexttile;
imagesc(x_km, y_km, log10(thickness_maps.thickness_mm), 'AlphaData', ~isnan(thickness_maps.thickness_mm));
axis image; set(gca, 'YDir', 'normal', 'Color', [0.85 0.85 0.85]);   % grey = no prediction
colormap(gca, parula);
colorbar_handle = colorbar; colorbar_handle.Label.String = 'log_{10} ash thickness (log_{10} mm)';
clim(log10_colour_limits);
xlabel('Easting (km)'); ylabel('Northing (km)');
title({'(c) Predicted thickness from DPM', 'grey = no prediction'});

nexttile;
imagesc(x_km, y_km, thickness_maps.flag);
axis image; set(gca, 'YDir', 'normal');
colormap(gca, [0.3 0.7 0.3; 1.0 0.8 0.2; 0.7 0.7 0.7; 0.8 0.2 0.2; 0 0 0; 0.55 0.25 0.75]);   % one colour per flag 0-5
clim([-0.5 5.5]);
colorbar_handle = colorbar('Ticks', 0:5, 'TickLabels', {'0 ok', '1 extrapolated DPM', '2 below floor', '3 above ceiling', '4 NaN input', '5 beyond calibration'});
colorbar_handle.Label.String = 'Prediction flag';
xlabel('Easting (km)'); ylabel('Northing (km)');
title({'(d) Prediction flags', ' '});

if ~isfolder(output_directory)
    mkdir(output_directory);
end
exportgraphics(figure_map, fullfile(output_directory, [output_basename '_05_map.png']), 'Resolution', 150);   % PNG at 150 dpi, cropped to the figure
exportgraphics(figure_map, fullfile(output_directory, [output_basename '_05_map.pdf']), 'ContentType', 'vector');   % PDF page = figure size (print -dpdf shrinks it onto US letter)

fprintf('Done. Output saved to %s\n', output_directory);

%% --- Local function: synthetic DPM map for section 5 ---
function [dpm_map, thickness_map_mm, x_km, y_km] = make_synthetic_dpm_map()
% MAKE_SYNTHETIC_DPM_MAP  A 150-by-150 map (0-30 km) of ash thinning away from a vent, and a
%   noisy DPM map generated from it with the same sigmoid as make_example_dpm_csv
%   (t50 = 30 mm, k = 2, floor 0.05, ceiling 0.90, logit noise 0.6). A lake is masked as NaN.
%   Returns dpm_map (dimensionless), thickness_map_mm (mm), and the x/y axes (km).
    random_stream = RandStream('mt19937ar', 'Seed', 7);
    x_km = linspace(0, 30, 150);
    y_km = linspace(0, 30, 150);
    [x_grid_km, y_grid_km] = meshgrid(x_km, y_km);
    vent_x_km = 8;
    vent_y_km = 15;
    % Elongated plume: thinning is slower downwind (to the east) than crosswind
    downwind_km = x_grid_km - vent_x_km;
    crosswind_km = y_grid_km - vent_y_km;
    plume_distance_km = sqrt((downwind_km / 2.2) .^ 2 + crosswind_km .^ 2);
    % Clamped to the calibrated range 0.1-499 mm: the flat region at 0.1 mm (log10 = -1) in
    % panel (b) is this clamp, not a physical feature
    thickness_map_mm = min(500 * exp(-plume_distance_km / 1.6), 499);   % mm, capped at the top isopach edge
    thickness_map_mm = max(thickness_map_mm, 0.1);                        % 0.1 mm = trace ash
    logit_noisy = 2.0 * (log10(thickness_map_mm) - log10(30)) + 0.6 * randn(random_stream, size(thickness_map_mm));
    dpm_map = 0.05 + (0.90 - 0.05) ./ (1 + exp(-logit_noisy));
    lake_mask = (x_grid_km - 22) .^ 2 + (y_grid_km - 6) .^ 2 < 3 ^ 2;  % e.g. water masked in a real DPM
    dpm_map(lake_mask) = NaN;
end

function write_example_csv(csv_file)
% WRITE_EXAMPLE_CSV  Writes the synthetic example CSV (columns dpm, thickness_mm), identical
%   to the one from make_example_dpm_csv.m: 7 classes x 500 pixels, thickness log-uniform in
%   each class (mm), DPM from the sigmoid t50 = 30 mm, k = 2, floor 0.05, ceiling 0.90 with
%   logit noise 0.6, random seed 11. Kept here (not a call to that script) because the script
%   starts with 'clear', which would wipe this example's variables.
    edges_mm = [0.1 1 5 10 50 100 200 500];
    pixels_per_class = 500;
    random_stream = RandStream('mt19937ar', 'Seed', 11);
    class_index = repelem((1:numel(edges_mm) - 1)', pixels_per_class);
    log10_lower = log10(edges_mm(class_index))';
    log10_upper = log10(edges_mm(class_index + 1))';
    thickness_mm = 10 .^ (log10_lower + (log10_upper - log10_lower) .* rand(random_stream, size(log10_lower)));
    logit_noisy = 2.0 * (log10(thickness_mm) - log10(30)) + 0.6 * randn(random_stream, size(thickness_mm));
    dpm = 0.05 + (0.90 - 0.05) ./ (1 + exp(-logit_noisy));
    writetable(table(dpm, thickness_mm), csv_file);
    fprintf('  Wrote %s (%d rows)\n', csv_file, numel(dpm));
end
