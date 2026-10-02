% Purpose:  Write a SYNTHETIC example CSV in the exact input format expected by
%           fit_thickness_from_dpm.m (2 columns: dpm, thickness_mm), run the full fit
%           at the real-data size (~50 pixels per class), and CHECK it:
%             (1) fitted curve vs the noise-free generator curve (approximate target)
%             (2) an independent test set (new random pixels, never seen by the fit):
%                 RMSE and prediction-interval coverage
%           Run this first to check your MATLAB setup. The numbers are made up; they
%           are NOT real observations.
% Inputs:   None
% Outputs:  example_dpm_thickness.csv, example_01_boxplot, example_02_fit,
%           example_03_bootstrap, example_04_plot2 (.png and .pdf) in output_directory
% Date:     2026-10-02

clear; clc; close all;

% --- File paths and parameters ---
output_directory      = '.';
example_csv_file      = fullfile(output_directory, 'example_dpm_thickness.csv');
isopach_edges_mm      = [0.1 1 5 10 50 100 200 500];   % isopach class boundaries (mm)
pixels_per_class_generated = 500;                      % synthetic pixels written to the CSV per class
pixels_per_class_used = 50;                            % pixels per class used in the fit (real data ~50)
true_steepness        = 2.0;                           % 1 / log10 mm
true_t50_mm           = 30;                            % mm, hypothetical
true_floor_dpm        = 0.05;                          % background decorrelation noise
true_ceiling_dpm      = 0.90;                          % response saturates below 1
spread_sigma_logit    = 0.6;                           % pixel-to-pixel spread (logit space keeps DPM in 0-1)
random_seed           = 11;                            % fixed for reproducibility
test_random_seed      = 99;                            % different seed: independent test pixels
test_pixels_per_class = 500;                           % independent test pixels per class

% --- Generate synthetic pixels and write the 2-column CSV ---
fprintf('[1/4] Generating synthetic example data (%d pixels per class)...\n', pixels_per_class_generated);
[dpm, thickness_mm] = generate_synthetic_pixels(isopach_edges_mm, pixels_per_class_generated, true_steepness, ...
    true_t50_mm, true_floor_dpm, true_ceiling_dpm, spread_sigma_logit, random_seed);
example_table = table(dpm, thickness_mm);     % header row is allowed and skipped by the function
writetable(example_table, example_csv_file);
fprintf('  Wrote %s (%d rows, columns: dpm, thickness_mm)\n', example_csv_file, height(example_table));

% --- Run the full fit at the real-data size ---
fprintf('[2/4] Running fit_thickness_from_dpm with %d pixels per class...\n', pixels_per_class_used);
fit_results = fit_thickness_from_dpm(example_csv_file, isopach_edges_mm, ...
    'MaxPixelsPerClass', pixels_per_class_used, ...
    'OutputDirectory', output_directory, 'OutputBasename', 'example');

% --- Check 1: fitted curve vs the generator's median curve ---
fprintf('[3/4] Check 1: fitted thickness vs the noise-free generator curve...\n');
check_dpm = [0.1 0.2 0.3 0.45 0.6 0.75 0.85]';
% APPROXIMATE target: the noise-free curve with the true parameters. The exact target of a fit of
% thickness on DPM is the median thickness GIVEN DPM, which is shifted from this curve by the
% sampling design (equal pixels per class, and classes span different log10 widths). So a few
% percent difference is expected even for a perfect fit; large differences are not.
true_thickness_mm = 10 .^ (log10(true_t50_mm) + (1 / true_steepness) * ...
    log((check_dpm - true_floor_dpm) ./ (true_ceiling_dpm - check_dpm)));
check_table = predict_thickness_from_dpm(fit_results, check_dpm);
check_table.noise_free_mm = true_thickness_mm;
check_table.ratio_fit_to_noise_free = check_table.thickness_mm ./ true_thickness_mm;
check_table.noise_free_in_ci = true_thickness_mm >= check_table.ci_low_mm & true_thickness_mm <= check_table.ci_high_mm;
disp(check_table(:, {'dpm', 'thickness_mm', 'ci_low_mm', 'ci_high_mm', 'noise_free_mm', 'ratio_fit_to_noise_free', 'noise_free_in_ci', 'note'}));

% --- Check 2: independent test pixels never seen by the fit ---
fprintf('[4/4] Check 2: %d independent test pixels per class (seed %d)...\n', test_pixels_per_class, test_random_seed);
[test_dpm, test_thickness_mm] = generate_synthetic_pixels(isopach_edges_mm, test_pixels_per_class, true_steepness, ...
    true_t50_mm, true_floor_dpm, true_ceiling_dpm, spread_sigma_logit, test_random_seed);
test_table = predict_thickness_from_dpm(fit_results, test_dpm);
predicted = isfinite(test_table.thickness_mm);         % flags 0, 1, 5 have a thickness; 2-4 do not
test_residual_log10_mm = log10(test_thickness_mm(predicted)) - log10(test_table.thickness_mm(predicted));
test_inside_pi = test_thickness_mm(predicted) >= test_table.pi_low_mm(predicted) & ...
                 test_thickness_mm(predicted) <= test_table.pi_high_mm(predicted);
fprintf('  Predicted %d of %d test pixels (%d below floor / above ceiling / NaN)\n', ...
    sum(predicted), numel(test_dpm), sum(~predicted));
fprintf('  Independent test: RMSE = %.3f log10 mm (OOB single-fit estimate %.3f), within x2 = %.0f%% (OOB %.0f%%)\n', ...
    sqrt(mean(test_residual_log10_mm .^ 2)), fit_results.oob.rmse_log10_mm, ...
    100 * mean(abs(test_residual_log10_mm) <= log10(2)), 100 * fit_results.oob.within_factor_2);
fprintf('  Independent test: %.0f%% prediction interval covers %.1f%% of predicted test pixels\n', ...
    100 * fit_results.prediction_interval.coverage, 100 * mean(test_inside_pi));
% Stricter: count pixels that got no prediction (below floor / above ceiling) as misses
fprintf('  ... or %.1f%% of ALL test pixels, counting the %d unpredicted ones as misses\n', ...
    100 * sum(test_inside_pi) / numel(test_dpm), sum(~predicted));
test_class = discretize(test_thickness_mm(predicted), isopach_edges_mm);
class_coverage = accumarray(test_class, test_inside_pi, [numel(isopach_edges_mm) - 1, 1], @mean);
for class_number = 1:numel(class_coverage)
    fprintf('    %-12s PI coverage %5.1f%%\n', sprintf('%g-%g', isopach_edges_mm(class_number), ...
        isopach_edges_mm(class_number + 1)), 100 * class_coverage(class_number));
end

fprintf('Done. Output saved to %s\n', output_directory);

% --- Local functions ---
function [dpm, thickness_mm] = generate_synthetic_pixels(edges_mm, pixels_per_class, steepness, t50_mm, ...
    floor_dpm, ceiling_dpm, spread_sigma_logit, seed)
% GENERATE_SYNTHETIC_PIXELS  Synthetic (dpm, thickness_mm) pixels from a known sigmoid.
%   Thickness is drawn log-uniformly inside each class (mm); DPM follows the logistic curve in
%   log10 thickness with Gaussian noise in logit space (sigma = spread_sigma_logit), which keeps
%   DPM between floor and ceiling. Uses its own random stream (seed), so it is reproducible.
%   Returns column vectors dpm (dimensionless) and thickness_mm (mm).
    random_stream = RandStream('mt19937ar', 'Seed', seed);
    number_of_classes = numel(edges_mm) - 1;
    class_index = repelem((1:number_of_classes)', pixels_per_class);
    log10_lower = log10(edges_mm(class_index))';
    log10_upper = log10(edges_mm(class_index + 1))';
    thickness_mm = 10 .^ (log10_lower + (log10_upper - log10_lower) .* rand(random_stream, size(log10_lower)));
    logit_noisy = steepness * (log10(thickness_mm) - log10(t50_mm)) + spread_sigma_logit * randn(random_stream, size(thickness_mm));
    dpm = floor_dpm + (ceiling_dpm - floor_dpm) ./ (1 + exp(-logit_noisy));
end
