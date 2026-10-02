function [prediction_table, prediction_maps] = predict_thickness_from_dpm(fit_results, dpm_values)
% PREDICT_THICKNESS_FROM_DPM  Ash thickness (mm) from DPM value(s), with a confidence
%                             interval of the curve and a single-pixel prediction interval.
%
% Usage:
%   fit_results = fit_thickness_from_dpm(data_matrix, edges_mm);
%   predict_thickness_from_dpm(fit_results, [0.2 0.4 0.6])            % no output: prints the table
%   prediction_table = predict_thickness_from_dpm(fit_results, dpm_values);
%   [~, maps] = predict_thickness_from_dpm(fit_results, dpm_image);   % DPM map in, thickness maps out
%
% Description:
%   Applies the inverse sigmoid fitted by fit_thickness_from_dpm (the fit carried forward,
%   'class_medians' or 'pixels'):
%       log10 t = log10 t50 + (1/k) * ln( (DPM - floor) / (ceiling - DPM) )
%   and returns, for every DPM value:
%
%   thickness_mm          point estimate (a typical, median-like thickness, NOT the mean)
%   ci_low/high_mm        95% CONFIDENCE interval of the curve: where the calibration curve
%                         itself lies (from the bootstrap curves). Use it for "what thickness
%                         does this DPM correspond to on average?"
%   pi_low/high_mm        PREDICTION interval for ONE pixel (coverage set in the fit, default
%                         95%): point estimate x the out-of-bag residual factors. Use it for
%                         "what is the thickness at THIS pixel?" Much wider than the CI.
%   flag / note           0 ok
%                         1 extrapolation: DPM outside the DPM range used to fit (values given)
%                         2 below floor: no detectable ash signal (thickness NaN)
%                         3 above ceiling: DPM saturated, thickness unbounded (thickness NaN)
%                         4 input DPM is NaN
%                         5 thickness outside the CALIBRATED range (thinner than the lowest or
%                           thicker than the highest isopach edge): value KEPT but not supported
%                           by data - treat it as "at least ~top edge" (or "at most ~lowest edge")
%
% Inputs:
%   fit_results : struct returned by fit_thickness_from_dpm
%   dpm_values  : DPM, dimensionless 0-1; vector or 2-D map (NaN allowed, e.g. masked pixels)
%
% Outputs:
%   prediction_table : table, one row per DPM value. NOT built (returned empty) when two
%                      outputs are requested, so large maps do not run out of memory
%   prediction_maps  : struct of arrays with the SAME SHAPE as dpm_values: thickness_mm,
%                      ci_low_mm, ci_high_mm, pi_low_mm, pi_high_mm, flag
%
% Scientific caveats:
%   - Valid only for DPM from the same kind of scene/pair as the calibration. DPM also
%     responds to non-ash surface change, so a high DPM does not prove thick ash.
%   - Intervals assume pixels are independent; real DPM pixels are spatially correlated,
%     so real uncertainty is larger.
%   - The PI has a constant width in log10 thickness; fit_results.prediction_interval
%     .class_coverage shows the classes where it covers less than intended.
%
% Date: 2026-10-02

% --- Validate inputs ---
if ~isstruct(fit_results) || ~isfield(fit_results, 'prediction_interval') || ~isfield(fit_results, 'bootstrap')
    error('predict_thickness_from_dpm:badFit', 'fit_results must come from fit_thickness_from_dpm.');
end
if ~isnumeric(dpm_values) || isempty(dpm_values)
    error('predict_thickness_from_dpm:badDpm', 'dpm_values must be a non-empty numeric array.');
end
input_shape = size(dpm_values);
dpm_column = double(dpm_values(:));
finite_dpm = dpm_column(isfinite(dpm_column));
if any(finite_dpm < 0 | finite_dpm > 1)
    error('predict_thickness_from_dpm:dpmRange', ...
        'dpm_values must be scaled 0 to 1 but found range [%.3g, %.3g]. Rescale first (e.g. divide by 100 or 255).', ...
        min(finite_dpm), max(finite_dpm));
end
number_of_values = numel(dpm_column);
fitted_params = fit_results.params;                  % [k, log10 t50, floor, ceiling]
floor_dpm = fitted_params(3);
ceiling_dpm = fitted_params(4);
dpm_clip_margin = fit_results.dpm_clip_margin;
observed_dpm_range = fit_results.observed_dpm_range;
if strcmp(fit_results.fit_target, 'class_medians')
    bootstrap_params = fit_results.bootstrap.median_fit.params;
else
    bootstrap_params = fit_results.bootstrap.pixel_fit.params;
end

% --- Flags ---
fprintf('[1/3] Flagging %d DPM value(s) (floor = %.3f, ceiling = %.3f, observed range %.3f-%.3f)...\n', ...
    number_of_values, floor_dpm, ceiling_dpm, observed_dpm_range(1), observed_dpm_range(2));
flag_code = zeros(number_of_values, 1);
flag_code(dpm_column < observed_dpm_range(1) | dpm_column > observed_dpm_range(2)) = 1;
flag_code(dpm_column <= floor_dpm + dpm_clip_margin) = 2;
flag_code(dpm_column >= ceiling_dpm - dpm_clip_margin) = 3;
flag_code(isnan(dpm_column)) = 4;
can_predict = flag_code <= 1;                        % ok or extrapolation

% --- Point estimate and prediction interval ---
fprintf('[2/3] Point estimate and %.0f%% single-pixel prediction interval...\n', 100 * fit_results.prediction_interval.coverage);
thickness_mm = nan(number_of_values, 1);
log10_point = inverse_sigmoid(dpm_column(can_predict), fitted_params, dpm_clip_margin);
thickness_mm(can_predict) = 10 .^ log10_point;
pi_factor = fit_results.prediction_interval.factor;  % [low high], multiplies the point estimate
pi_low_mm = thickness_mm * pi_factor(1);
pi_high_mm = thickness_mm * pi_factor(2);

% --- Confidence interval of the curve from the bootstrap curves (chunked for large maps) ---
fprintf('[3/3] 95%% confidence interval from %d bootstrap curves...\n', size(bootstrap_params, 1));
t0 = tic;
ci_low_mm = nan(number_of_values, 1);
ci_high_mm = nan(number_of_values, 1);
predict_index = find(can_predict);
chunk_size = 2000;                                   % keeps the chunk-by-bootstrap matrix small (2000 x B)
for chunk_start = 1:chunk_size:numel(predict_index)
    chunk_index = predict_index(chunk_start:min(chunk_start + chunk_size - 1, numel(predict_index)));
    % chunk-by-B matrix of log10 thickness, one column per bootstrap curve
    chunk_log10 = inverse_sigmoid(dpm_column(chunk_index), bootstrap_params', dpm_clip_margin);
    chunk_limits = quantile(chunk_log10, [0.025 0.975], 2);
    ci_low_mm(chunk_index) = 10 .^ chunk_limits(:, 1);
    ci_high_mm(chunk_index) = 10 .^ chunk_limits(:, 2);
end
fprintf('  Done in %.1fs\n', toc(t0));
% Flag 5 (after prediction): thickness outside the calibrated isopach range. The value is kept
% (user choice) but is NOT supported by the calibration data. Takes precedence over flag 1.
calibrated_range_mm = fit_results.calibrated_thickness_range_mm;
outside_calibration = can_predict & (thickness_mm > calibrated_range_mm(2) | thickness_mm < calibrated_range_mm(1));
flag_code(outside_calibration) = 5;
fprintf('  Flags: %d ok, %d extrapolated DPM, %d below floor, %d above ceiling, %d NaN input, %d outside calibrated %g-%g mm\n', ...
    sum(flag_code == 0), sum(flag_code == 1), sum(flag_code == 2), sum(flag_code == 3), sum(flag_code == 4), ...
    sum(flag_code == 5), calibrated_range_mm(1), calibrated_range_mm(2));

% --- Outputs ---
if nargout >= 2
    prediction_maps = struct( ...
        'thickness_mm', reshape(thickness_mm, input_shape), ...
        'ci_low_mm', reshape(ci_low_mm, input_shape), ...
        'ci_high_mm', reshape(ci_high_mm, input_shape), ...
        'pi_low_mm', reshape(pi_low_mm, input_shape), ...
        'pi_high_mm', reshape(pi_high_mm, input_shape), ...
        'flag', reshape(flag_code, input_shape));
end
prediction_table = table();                          % empty in map mode (two outputs): a 25M-row table would not fit in memory
if nargout <= 1
    note_text = {'ok'; 'extrapolation: DPM outside fitted range'; 'below floor: no detectable ash signal'; ...
        'above ceiling: DPM saturated'; 'input DPM is NaN'; 'outside calibrated thickness range: not supported'};
    prediction_table = table(dpm_column, thickness_mm, ci_low_mm, ci_high_mm, pi_low_mm, pi_high_mm, flag_code, ...
        note_text(flag_code + 1), ...
        'VariableNames', {'dpm', 'thickness_mm', 'ci_low_mm', 'ci_high_mm', 'pi_low_mm', 'pi_high_mm', 'flag', 'note'});
end
if nargout == 0
    disp(prediction_table);
    fprintf('CI = where the calibration curve lies; PI = where a single pixel''s thickness could lie.\n');
end
fprintf('Done.\n');
end

% --- Local functions ---
function log10_thickness = inverse_sigmoid(dpm_values, params, dpm_clip_margin)
% INVERSE_SIGMOID  log10 ash thickness (log10 mm) from DPM; same model as fit_thickness_from_dpm.
%   dpm_values : column vector of DPM (dimensionless), strictly between floor and ceiling
%   params     : 4-by-1 or 4-by-B [k; log10 t50; floor; ceiling] (one column per curve)
%   dpm_clip_margin : DPM beyond a curve's floor/ceiling is clipped this far inside them
%                     (same margin as in the fit, DPM units)
%   Returns N-by-B. Each bootstrap curve has its own floor/ceiling, so values outside them
%   get a finite but extreme thickness, exactly as in fit_thickness_from_dpm.
    if isrow(params)
        params = params';
    end
    steepness = params(1, :);
    log10_t50 = params(2, :);
    floor_dpm = params(3, :);
    ceiling_dpm = params(4, :);
    clipped_dpm = min(max(dpm_values, floor_dpm + dpm_clip_margin), ceiling_dpm - dpm_clip_margin);
    log10_thickness = log10_t50 + (1 ./ steepness) .* log((clipped_dpm - floor_dpm) ./ (ceiling_dpm - clipped_dpm));
end
