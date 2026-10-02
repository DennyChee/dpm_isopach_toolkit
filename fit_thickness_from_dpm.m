function fit_results = fit_thickness_from_dpm(dpm_thickness_data, isopach_edges_mm, varargin)
% FIT_THICKNESS_FROM_DPM  Predict ash thickness from DPM with an inverse sigmoid,
%                         fitted directly in log10 thickness.
%
% Usage:
%   fit_results = fit_thickness_from_dpm(data_matrix, [0.1 1 5 10 50 100 200 500]);
%   fit_results = fit_thickness_from_dpm('dpm_thickness.csv', [0.1 1 5 10 50 100 200 500], ...
%                     'OutputDirectory', 'figures');
%
% Description:
%   Plot 1: box-and-whisker of DPM per isopach thickness class
%           (DPM on x, thickness class on y).
%   Fit:    inverse sigmoid fitted DIRECTLY in log10 thickness (least squares on log10 mm):
%             log10 t = log10 t50 + (1/k) * ln( (DPM - floor) / (ceiling - DPM) )
%           Two versions are fitted; FitTarget picks the one carried forward:
%             'class_medians' : one point per class = (median DPM, median log10 thickness)
%                               - the medians of the box plot; robust, equal weight per class
%             'pixels'        : every pixel, using its own thickness
%           For fixed floor/ceiling this is a straight line in the logit term, so
%           log10 t50 and 1/k are solved exactly and only floor/ceiling are searched.
%           DPM at/beyond floor or ceiling is clipped DpmClipMargin inside them.
%   Bootstrap: out-of-bag (OOB). Each round resamples pixels with replacement WITHIN each
%           class, refits both versions, and predicts the ~37% of pixels not drawn
%           (out-of-bag = hold-out for that round). Gives parameter intervals, a 95%
%           confidence band of the curve (over the observed DPM range only), and honest
%           hold-out scores in log10 thickness, overall and per class.
%   Plot 2: final calibration plot - box-and-whisker on a log thickness axis, best-fit
%           curve, 95% confidence band of the curve (bootstrap), single-pixel prediction
%           interval (from pooled out-of-bag residuals, constant factor), hold-out scores.
%
% ---------------------------------------------------------------------------
% INPUTS
% ---------------------------------------------------------------------------
%   dpm_thickness_data : N-by-2 numeric matrix, ONE ROW PER PIXEL
%                          column 1 = DPM value, dimensionless, scaled 0 to 1
%                          column 2 = ash thickness at that pixel, in mm
%                        OR the path to a CSV file with those same two columns
%                        (a text header row is allowed and skipped).
%   isopach_edges_mm   : increasing vector of class boundaries in mm, all > 0,
%                        e.g. [0.1 1 5 10 50 100 200 500] gives 7 classes.
%                        Used for the box plot and to stratify the bootstrap.
%                        Pixels outside [first edge, last edge] are dropped.
%
% ---------------------------------------------------------------------------
% OPTIONS (name-value pairs) - defaults are set in the DEFAULT SETTINGS block
% at the top of the code; edit them there or override them in the call
% ---------------------------------------------------------------------------
%   'OutputDirectory'    : where figures are saved
%   'OutputBasename'     : file name prefix without extension
%   'MaxPixelsPerClass'  : randomly keep at most this many pixels per class so
%                          big classes do not dominate
%   'MinPixelsPerClass'  : STOP with an error if any class has fewer pixels than this
%                          (the message lists the classes and where to edit)
%   'WarnPixelsPerClass' : warn, but continue, for classes with fewer pixels than this
%   'RandomSeed'         : seed for all random steps, for reproducibility
%   'MakePlot'           : true/false; false skips figures
%   'NumBootstrap'       : number of out-of-bag bootstrap rounds (>= 50)
%   'Coverage'           : coverage of the single-pixel prediction interval (e.g. 0.95)
%   'SaturationDpm'      : DPM at/above this counts as saturated (e.g. 0.99)
%   'SaturationWarnFraction', 'SaturationStopFraction' : warn / STOP when more than this
%                          fraction of any class is saturated
%   'FitTarget'          : 'class_medians' or 'pixels' (see Fit above)
%   'DpmClipMargin'      : DPM at/below floor or at/above ceiling is clipped this far
%                          inside them before the logit (dimensionless, DPM units)
%
% Output (struct, grows in later steps):
%   dpm, thickness_mm, class_index : the pixels kept (column vectors)
%   class_table                    : per-class edges, pixel count, median DPM, median thickness,
%                                    saturated pixel count, unpredictable fraction
%   calibrated_thickness_range_mm  : [lowest edge, highest edge]; predictions outside = flag 5
%   steepness_per_log10_mm, t50_mm, floor_dpm, ceiling_dpm : best-fit parameters
%   params                         : [k, log10 t50, floor, ceiling] of the fit carried forward
%   fit_target                     : which fit was carried forward
%   median_fit, pixel_fit          : params and in-sample scores of both fits
%   bootstrap                      : per fit (median_fit, pixel_fit): bootstrap params,
%                                    parameter_ci_95, confidence band, OOB scores (overall,
%                                    per round, per class), per-pixel OOB predictions
%   parameter_ci_95, oob           : 95% parameter intervals and OOB scores of the
%                                    carried-forward fit (oob.rmse_log10_mm = single fit, median
%                                    over rounds; oob.bagged_* = averaged OOB predictions)
%   prediction_interval            : single-pixel PI: log10_mm [low high] to add to the
%                                    predicted log10 thickness, factor = 10.^log10_mm,
%                                    and its coverage per class
%   observed_dpm_range             : [min max] DPM of the pixels used; outside it = extrapolation
%   in_sample                      : RMSE (log10 mm), R^2 (log10 thickness), fraction
%                                    within x2 - IN-SAMPLE, optimistic; see out-of-bag later
%
% Date: 2026-10-02

% ===========================================================================
% DEFAULT SETTINGS - edit these to change the defaults
% (any of them can also be overridden per call with a name-value pair)
% ===========================================================================
default_output_directory      = '.';
default_output_basename       = 'dpm_thickness';
default_max_pixels_per_class  = 5000;   % cap per class; real data has ~50 per class, so the cap rarely bites
default_min_pixels_per_class  = 10;     % STOP with an error if any class has fewer pixels than this
default_warn_pixels_per_class = 30;     % warn (but continue) if any class has fewer pixels than this
default_random_seed           = 42;
default_make_plot             = true;
default_dpm_clip_margin       = 0.005;  % DPM units; pixels beyond floor/ceiling are clipped this far inside
default_number_of_bootstraps  = 1000;   % out-of-bag bootstrap rounds (more = smoother intervals, slower)
default_prediction_coverage   = 0.95;   % coverage of the single-pixel prediction interval
default_fit_target            = 'class_medians';   % 'class_medians' (one point per box) or 'pixels' (every pixel)
default_saturation_dpm        = 0.99;   % DPM at/above this counts as SATURATED (the 'ruler has run out')
default_saturation_warn_fraction = 0.10;   % warn if more than this fraction of a class is saturated
default_saturation_stop_fraction = 0.50;   % STOP if more than this fraction of a class is saturated (its median is saturated)
% ===========================================================================

% --- Parse options ---
option_parser = inputParser;
addRequired(option_parser, 'dpm_thickness_data', @(value) isnumeric(value) || ischar(value) || isstring(value));
addRequired(option_parser, 'isopach_edges_mm', @(value) isnumeric(value) && isvector(value));
addParameter(option_parser, 'OutputDirectory', default_output_directory, @(value) ischar(value) || isstring(value));
addParameter(option_parser, 'OutputBasename', default_output_basename, @(value) ischar(value) || isstring(value));
addParameter(option_parser, 'MaxPixelsPerClass', default_max_pixels_per_class, @(value) isnumeric(value) && isscalar(value));
addParameter(option_parser, 'MinPixelsPerClass', default_min_pixels_per_class, @(value) isnumeric(value) && isscalar(value));
addParameter(option_parser, 'WarnPixelsPerClass', default_warn_pixels_per_class, @(value) isnumeric(value) && isscalar(value));
addParameter(option_parser, 'RandomSeed', default_random_seed, @(value) isnumeric(value) && isscalar(value));
addParameter(option_parser, 'MakePlot', default_make_plot, @(value) islogical(value) || isnumeric(value));
addParameter(option_parser, 'NumBootstrap', default_number_of_bootstraps, @(value) isnumeric(value) && isscalar(value) && value >= 50);
addParameter(option_parser, 'Coverage', default_prediction_coverage, @(value) isnumeric(value) && isscalar(value) && value > 0.5 && value < 1);
addParameter(option_parser, 'SaturationDpm', default_saturation_dpm, @(value) isnumeric(value) && isscalar(value) && value > 0.5 && value <= 1);
addParameter(option_parser, 'SaturationWarnFraction', default_saturation_warn_fraction, @(value) isnumeric(value) && isscalar(value) && value >= 0 && value <= 1);
addParameter(option_parser, 'SaturationStopFraction', default_saturation_stop_fraction, @(value) isnumeric(value) && isscalar(value) && value >= 0 && value <= 1);
addParameter(option_parser, 'FitTarget', default_fit_target, @(value) any(strcmp(value, {'class_medians', 'pixels'})));
addParameter(option_parser, 'DpmClipMargin', default_dpm_clip_margin, @(value) isnumeric(value) && isscalar(value) && value > 0 && value < 0.1);
parse(option_parser, dpm_thickness_data, isopach_edges_mm, varargin{:});
output_directory      = char(option_parser.Results.OutputDirectory);
output_basename       = char(option_parser.Results.OutputBasename);
max_pixels_per_class  = option_parser.Results.MaxPixelsPerClass;
min_pixels_per_class  = option_parser.Results.MinPixelsPerClass;
warn_pixels_per_class = option_parser.Results.WarnPixelsPerClass;
random_seed           = option_parser.Results.RandomSeed;
make_plot             = logical(option_parser.Results.MakePlot);
dpm_clip_margin       = option_parser.Results.DpmClipMargin;
fit_target            = char(option_parser.Results.FitTarget);
number_of_bootstraps  = option_parser.Results.NumBootstrap;
prediction_coverage   = option_parser.Results.Coverage;
saturation_dpm        = option_parser.Results.SaturationDpm;
saturation_warn_fraction = option_parser.Results.SaturationWarnFraction;
saturation_stop_fraction = option_parser.Results.SaturationStopFraction;
isopach_edges_mm     = isopach_edges_mm(:)';   % force a row vector

% --- Load and validate input data ---
if isnumeric(dpm_thickness_data)
    fprintf('[1/5] Using the matrix supplied in the workspace...\n');
    data_matrix = double(dpm_thickness_data);
else
    csv_file = char(dpm_thickness_data);
    fprintf('[1/5] Loading %s...\n', csv_file);
    if ~isfile(csv_file)
        error('fit_thickness_from_dpm:fileNotFound', 'Input file not found: %s', csv_file);
    end
    data_matrix = readmatrix(csv_file);   % skips a text header row automatically
end
if size(data_matrix, 2) ~= 2
    error('fit_thickness_from_dpm:badShape', ...
        'Data must have exactly 2 columns [dpm, thickness_mm] but has %d.', size(data_matrix, 2));
end
fprintf('  Loaded: shape=%s, class=%s\n', mat2str(size(data_matrix)), class(data_matrix));
if any(diff(isopach_edges_mm) <= 0) || any(isopach_edges_mm <= 0) || numel(isopach_edges_mm) < 4
    error('fit_thickness_from_dpm:badEdges', ...
        'isopach_edges_mm must be increasing, all > 0 (use e.g. 0.1, not 0), with at least 3 classes (4 edges).');
end

dpm_data = data_matrix(:, 1);
thickness_data_mm = data_matrix(:, 2);
valid_row_mask = isfinite(dpm_data) & isfinite(thickness_data_mm);
if any(~valid_row_mask)
    fprintf('  Dropped %d rows with NaN/empty values\n', sum(~valid_row_mask));
end
dpm_data = dpm_data(valid_row_mask);
thickness_data_mm = thickness_data_mm(valid_row_mask);

% Range check that catches the most common mistake (DPM in percent or 8-bit, or swapped columns)
if min(dpm_data) < 0 || max(dpm_data) > 1
    error('fit_thickness_from_dpm:dpmRange', ...
        ['Column 1 (dpm) must be scaled 0 to 1 but found range [%.3g, %.3g]. ' ...
         'Rescale first (e.g. divide by 100 or 255). Check the columns are not swapped.'], min(dpm_data), max(dpm_data));
end

% --- Assign pixels to isopach classes ---
fprintf('  Assigning %d pixels to isopach classes...\n', numel(dpm_data));
class_number_of_pixel = discretize(thickness_data_mm, isopach_edges_mm);   % NaN outside the edges
inside_edges_mask = ~isnan(class_number_of_pixel);
if any(~inside_edges_mask)
    fprintf('  Dropped %d pixels outside the edge range [%g, %g] mm\n', ...
        sum(~inside_edges_mask), isopach_edges_mm(1), isopach_edges_mm(end));
end
dpm_data = dpm_data(inside_edges_mask);
thickness_data_mm = thickness_data_mm(inside_edges_mask);
class_number_of_pixel = class_number_of_pixel(inside_edges_mask);

% Randomly cap the number of pixels per class so that no class dominates
% Private random stream: reproducible, and does not reset the caller's global rng
subsample_stream = RandStream('mt19937ar', 'Seed', random_seed);
keep_pixel_mask = false(size(dpm_data));
for class_number = unique(class_number_of_pixel)'
    pixel_list = find(class_number_of_pixel == class_number);
    if numel(pixel_list) > max_pixels_per_class
        pixel_list = pixel_list(randperm(subsample_stream, numel(pixel_list), max_pixels_per_class));
    end
    keep_pixel_mask(pixel_list) = true;
end
dpm_data = dpm_data(keep_pixel_mask);
thickness_data_mm = thickness_data_mm(keep_pixel_mask);
class_number_of_pixel = class_number_of_pixel(keep_pixel_mask);

% --- Data-size check: STOP if any class is too small ---
% A class with too few pixels gives a meaningless box, almost no weight in the fit and
% (with ~37% out-of-bag per bootstrap round) almost nothing to test on. Classes are NOT
% silently dropped, because that would quietly change the thickness range of the fit.
pixel_count_all_classes = accumarray(class_number_of_pixel, 1, [numel(isopach_edges_mm) - 1, 1]);
too_small_class_list = find(pixel_count_all_classes < min_pixels_per_class);
if ~isempty(too_small_class_list)
    class_report = '';
    for class_number = 1:numel(pixel_count_all_classes)
        flag_text = '';
        if pixel_count_all_classes(class_number) < min_pixels_per_class
            flag_text = '   <-- TOO SMALL';
        end
        class_report = [class_report, sprintf('    %g-%g mm: %d pixels%s\n', isopach_edges_mm(class_number), ...
            isopach_edges_mm(class_number + 1), pixel_count_all_classes(class_number), flag_text)]; %#ok<AGROW>
    end
    cap_note = '';
    if max_pixels_per_class < min_pixels_per_class
        cap_note = sprintf(['  * MaxPixelsPerClass (%d) is below MinPixelsPerClass: raise default_max_pixels_per_class\n' ...
                            '    at %s, or pass ''MaxPixelsPerClass'', N in the call.\n'], ...
                            max_pixels_per_class, settings_location('default_max_pixels_per_class'));
    end
    % Suggested edges: merge each too-small class into a neighbour by removing the
    % shared inner edge (the edge above it, or below it for the thickest class)
    number_of_all_classes = numel(pixel_count_all_classes);
    edges_to_remove = zeros(size(too_small_class_list));
    for list_position = 1:numel(too_small_class_list)
        small_class = too_small_class_list(list_position);
        if small_class < number_of_all_classes
            edges_to_remove(list_position) = small_class + 1;   % merge with the class above (thicker)
        else
            edges_to_remove(list_position) = small_class;       % thickest class: merge with the class below
        end
    end
    suggested_edges_mm = isopach_edges_mm;
    suggested_edges_mm(unique(edges_to_remove)) = [];
    error('fit_thickness_from_dpm:dataTooSmall', ...
        ['Data too small: %d class(es) have fewer than %d pixels (MinPixelsPerClass). Stopping.\n%s' ...
         'Fix one of these:\n' ...
         '  * Merge thin classes: remove an edge from isopach_edges_mm in YOUR call\n' ...
         '    Suggested: %s -> %s  (re-run; repeat if a merged class is still too small)\n' ...
         '  * Lower the limit: edit default_min_pixels_per_class at %s,\n' ...
         '    or pass ''MinPixelsPerClass'', N in the call (fit and hold-out scores become less reliable).\n%s'], ...
        numel(too_small_class_list), min_pixels_per_class, class_report, ...
        mat2str(isopach_edges_mm), mat2str(suggested_edges_mm), ...
        settings_location('default_min_pixels_per_class'), cap_note);
end
classes_to_keep = (1:numel(pixel_count_all_classes))';   % every class passed the check
[~, class_index_data] = ismember(class_number_of_pixel, classes_to_keep);   % 0 for dropped classes
dpm_data = dpm_data(class_index_data > 0);
thickness_data_mm = thickness_data_mm(class_index_data > 0);
class_index_data = class_index_data(class_index_data > 0);
number_of_classes = numel(classes_to_keep);
class_lower_mm = isopach_edges_mm(classes_to_keep)';
class_upper_mm = isopach_edges_mm(classes_to_keep + 1)';
pixel_count_per_class = accumarray(class_index_data, 1);
class_median_dpm = accumarray(class_index_data, dpm_data, [], @median);

fprintf('  %-14s %8s %12s\n', 'Class (mm)', 'Pixels', 'Median DPM');
for class_number = 1:number_of_classes
    fprintf('  %-14s %8d %12.3f\n', sprintf('%g-%g', class_lower_mm(class_number), class_upper_mm(class_number)), ...
        pixel_count_per_class(class_number), class_median_dpm(class_number));
end
pixel_count_ratio = max(pixel_count_per_class) / min(pixel_count_per_class);
if pixel_count_ratio > 3
    warning('fit_thickness_from_dpm:unequalClasses', ...
        ['Pixel counts differ by %.1fx between classes; larger classes weigh more in the fit. ' ...
         'Lower MaxPixelsPerClass to balance them.'], pixel_count_ratio);
end
small_class_list = find(pixel_count_per_class < warn_pixels_per_class);
if ~isempty(small_class_list)
    % With n pixels, each bootstrap round leaves only ~0.37*n of them out-of-bag for testing
    small_class_text = strjoin(arrayfun(@(class_number) sprintf('%g-%g mm (%d px)', class_lower_mm(class_number), ...
        class_upper_mm(class_number), pixel_count_per_class(class_number)), small_class_list', 'UniformOutput', false), ', ');
    warning('fit_thickness_from_dpm:smallClass', ...
        ['%d class(es) have fewer than %d pixels: %s. Their boxes, fit weight and hold-out scores are unreliable. ' ...
         'Continuing. To merge classes, remove an edge from isopach_edges_mm in your call; to change this threshold, ' ...
         'edit default_warn_pixels_per_class at %s.'], ...
        numel(small_class_list), warn_pixels_per_class, small_class_text, settings_location('default_warn_pixels_per_class'));
end

% --- Saturation check: pixels whose DPM has 'run out of ruler' ---
% DPM at/above saturation_dpm only says "a lot of change", not how much. They stay in the fit
% (a class median is unaffected while fewer than half of its pixels are saturated), but they
% get no thickness at prediction time and are left out of the hold-out scores.
saturated_count_per_class = accumarray(class_index_data, double(dpm_data >= saturation_dpm), [number_of_classes 1]);
saturated_fraction_per_class = saturated_count_per_class ./ pixel_count_per_class;
if any(saturated_count_per_class > 0)
    fprintf('  Saturated pixels (DPM >= %.3g) per class:\n', saturation_dpm);
    for class_number = find(saturated_count_per_class > 0)'
        fprintf('    %-12s %4d of %4d (%.0f%%)\n', sprintf('%g-%g', class_lower_mm(class_number), class_upper_mm(class_number)), ...
            saturated_count_per_class(class_number), pixel_count_per_class(class_number), 100 * saturated_fraction_per_class(class_number));
    end
end
saturated_class_text = @(class_list) strjoin(arrayfun(@(class_number) sprintf('%g-%g mm (%.0f%%)', ...
    class_lower_mm(class_number), class_upper_mm(class_number), 100 * saturated_fraction_per_class(class_number)), ...
    class_list(:)', 'UniformOutput', false), ', ');
stop_class_list = find(saturated_fraction_per_class > saturation_stop_fraction);
if ~isempty(stop_class_list)
    error('fit_thickness_from_dpm:saturated', ...
        ['DPM saturated: more than %.0f%% of the pixels have DPM >= %.3g in class(es) %s. Stopping.\n' ...
         'Their class median is itself saturated, so the curve cannot be placed there.\n' ...
         'Fix one of these:\n' ...
         '  * Check the DPM product is not clipped at 1 (or 255/100 before scaling).\n' ...
         '  * Merge the saturated classes into one top class (remove edges from isopach_edges_mm in YOUR call),\n' ...
         '    or drop them by lowering the last edge: the curve then stops below them.\n' ...
         '  * Change the limits: default_saturation_dpm at %s,\n' ...
         '    default_saturation_stop_fraction at %s (raising it is not recommended).'], ...
        100 * saturation_stop_fraction, saturation_dpm, saturated_class_text(stop_class_list), ...
        settings_location('default_saturation_dpm'), settings_location('default_saturation_stop_fraction'));
end
warn_class_list = find(saturated_fraction_per_class > saturation_warn_fraction);
if ~isempty(warn_class_list)
    warning('fit_thickness_from_dpm:partlySaturated', ...
        ['More than %.0f%% of the pixels have saturated DPM (>= %.3g) in class(es) %s. The fit is shakier there: ' ...
         'saturated pixels say "thick" but not how thick. Continuing. Thresholds: default_saturation_dpm at %s, ' ...
         'default_saturation_warn_fraction at %s.'], 100 * saturation_warn_fraction, saturation_dpm, ...
        saturated_class_text(warn_class_list), settings_location('default_saturation_dpm'), ...
        settings_location('default_saturation_warn_fraction'));
end

% --- Plot 1: box-and-whisker of DPM per thickness class ---
if make_plot
    fprintf('[2/5] Plot 1: box-and-whisker, saving to %s...\n', output_directory);
    if ~isfolder(output_directory)
        mkdir(output_directory);
    end
    class_labels = arrayfun(@(lower_mm, upper_mm, pixel_count) sprintf('%g-%g (n=%d)', lower_mm, upper_mm, pixel_count), ...
        class_lower_mm, class_upper_mm, pixel_count_per_class, 'UniformOutput', false);

    figure_box = figure('Position', [100 100 800 600]);
    % Horizontal boxes: DPM on x, one box per thickness class on y (thin at the bottom, thick at the top)
    boxchart(class_index_data, dpm_data, 'Orientation', 'horizontal', ...
        'BoxFaceColor', [0.4 0.4 0.4], 'MarkerColor', [0.4 0.4 0.4]);
    yticks(1:number_of_classes);
    yticklabels(class_labels);
    ylim([0.5, number_of_classes + 0.5]);
    xlim([0 1]);
    xlabel('DPM (probability of coherence change, dimensionless, 0-1)');
    ylabel('Ash thickness, isopach class (mm)');
    title(sprintf('DPM per ash-thickness class (%d pixels)', numel(dpm_data)));
    grid on;
    axtoolbar(gca, {});   % hide the axes toolbar so it is not printed into the figure

    exportgraphics(figure_box, fullfile(output_directory, [output_basename '_01_boxplot.png']), 'Resolution', 150);   % PNG at 150 dpi, cropped to the figure
    exportgraphics(figure_box, fullfile(output_directory, [output_basename '_01_boxplot.pdf']), 'ContentType', 'vector');   % PDF page = figure size (print -dpdf shrinks it onto US letter)
end

% --- Fit the inverse sigmoid: (a) to the class medians, (b) to all pixels ---
% Both use the same model and least squares in log10 thickness. FitTarget picks which
% one is carried forward (bootstrap, prediction); the other is kept for comparison.
fprintf('[3/5] Fitting inverse sigmoid in log10 thickness (class medians and all pixels)...\n');
log10_thickness_data = log10(thickness_data_mm);   % log10 mm, each pixel's own thickness
% One point per class: median DPM (the line in each box) and median log10 thickness of the
% class's pixels (equals the class value if every pixel in the class has the same thickness)
class_median_log10_thickness = accumarray(class_index_data, log10_thickness_data, [], @median);
t0 = tic;
[median_fit_params, median_fit_exit_flag] = fit_inverse_sigmoid(class_median_dpm, class_median_log10_thickness, dpm_clip_margin, []);
[pixel_fit_params, pixel_fit_exit_flag] = fit_inverse_sigmoid(dpm_data, log10_thickness_data, dpm_clip_margin, []);
fprintf('  Done in %.2fs (exit flags: medians=%d, pixels=%d; 1 = converged)\n', toc(t0), median_fit_exit_flag, pixel_fit_exit_flag);
% No valid fit = no floor/ceiling for which thickness INCREASES with DPM (slope 1/k > 0)
if any(~isfinite(median_fit_params)) || any(~isfinite(pixel_fit_params))
    error('fit_thickness_from_dpm:noFit', ...
        ['No valid inverse-sigmoid fit (medians fit valid: %d, pixels fit valid: %d): the data show no ' ...
         'increasing relationship between DPM and thickness. Check that column 1 is DPM (0-1) and ' ...
         'column 2 is thickness in mm, and that the DPM pair brackets the ash fall.'], ...
        all(isfinite(median_fit_params)), all(isfinite(pixel_fit_params)));
end
fprintf('  Median fit uses %d points for 4 parameters (%d degrees of freedom left)\n', ...
    number_of_classes, number_of_classes - 4);
if number_of_classes < 5 && strcmp(fit_target, 'class_medians')
    % 4 parameters from <= 4 medians: the curve passes exactly through (or is underdetermined by)
    % the points, so its bootstrap band and hold-out scores would be meaningless
    error('fit_thickness_from_dpm:fewMedians', ...
        ['Only %d classes: the medians fit needs at least 5 (4 parameters, one point per class). Stopping.\n' ...
         'Fix one of these:\n' ...
         '  * Use more isopach classes (add edges to isopach_edges_mm in YOUR call), or\n' ...
         '  * Fit to all pixels instead: edit default_fit_target at %s,\n' ...
         '    or pass ''FitTarget'', ''pixels'' in the call.'], ...
        number_of_classes, settings_location('default_fit_target'));
end

% Scores of BOTH fits on all pixels (in-sample: these pixels were used to fit, so optimistic)
% Only pixels with DPM strictly inside each fit's floor/ceiling: the others get no thickness
% at prediction time (flags 2/3), so scoring their clipped values would be misleading
median_fit_predictable = dpm_data > median_fit_params(3) + dpm_clip_margin & dpm_data < median_fit_params(4) - dpm_clip_margin;
pixel_fit_predictable = dpm_data > pixel_fit_params(3) + dpm_clip_margin & dpm_data < pixel_fit_params(4) - dpm_clip_margin;
median_fit_scores = score_log10_prediction(predict_log10_thickness(dpm_data(median_fit_predictable), median_fit_params, dpm_clip_margin), ...
    log10_thickness_data(median_fit_predictable));
pixel_fit_scores = score_log10_prediction(predict_log10_thickness(dpm_data(pixel_fit_predictable), pixel_fit_params, dpm_clip_margin), ...
    log10_thickness_data(pixel_fit_predictable));
median_fit_residual_at_medians = class_median_log10_thickness - ...
    predict_log10_thickness(class_median_dpm, median_fit_params, dpm_clip_margin);

fprintf('  %-34s %12s %12s\n', '', 'Medians', 'Pixels');
fprintf('  %-34s %12.2f %12.2f\n', 't50 (mm)', 10 ^ median_fit_params(2), 10 ^ pixel_fit_params(2));
fprintf('  %-34s %12.3f %12.3f\n', 'steepness k (per log10 mm)', median_fit_params(1), pixel_fit_params(1));
fprintf('  %-34s %12.3f %12.3f\n', 'floor (DPM)', median_fit_params(3), pixel_fit_params(3));
fprintf('  %-34s %12.3f %12.3f\n', 'ceiling (DPM)', median_fit_params(4), pixel_fit_params(4));
fprintf('  %-34s %12.3f %12.3f\n', 'Pixel RMSE (log10 mm, in-sample)', median_fit_scores.rmse_log10_mm, pixel_fit_scores.rmse_log10_mm);
fprintf('  %-34s %12.3f %12.3f\n', 'Pixel R^2 (log10 t, in-sample)', median_fit_scores.r_squared, pixel_fit_scores.r_squared);
fprintf('  %-34s %11.0f%% %11.0f%%\n', 'Pixels within x2 (in-sample)', 100 * median_fit_scores.within_factor_2, 100 * pixel_fit_scores.within_factor_2);
fprintf('  Median fit misfit at the class medians (log10 mm): %s\n', mat2str(round(median_fit_residual_at_medians', 3)));
fprintf('  (in-sample scores are optimistic; honest hold-out scores come from the out-of-bag bootstrap)\n');

% --- Select the fit carried forward ---
if strcmp(fit_target, 'class_medians')
    fitted_params = median_fit_params;
    exit_flag = median_fit_exit_flag;
    in_sample_scores = median_fit_scores;
    fit_target_label = 'class medians';
else
    fitted_params = pixel_fit_params;
    exit_flag = pixel_fit_exit_flag;
    in_sample_scores = pixel_fit_scores;
    fit_target_label = 'all pixels';
end
fprintf('  Carried forward: fit to %s (FitTarget = ''%s'')\n', fit_target_label, fit_target);
if exit_flag ~= 1
    warning('fit_thickness_from_dpm:notConverged', 'Fit did not converge; treat the parameters with caution.');
end
fitted_steepness = fitted_params(1);
fitted_t50_mm = 10 ^ fitted_params(2);
fitted_floor_dpm = fitted_params(3);
fitted_ceiling_dpm = fitted_params(4);

% Floor/ceiling stuck at their limits means the data do not constrain them:
% the curve beyond the observed DPM range is then extrapolation
observed_dpm_range = [min(dpm_data), max(dpm_data)];
if fitted_ceiling_dpm >= 1 - 1e-3
    warning('fit_thickness_from_dpm:ceilingAtLimit', ...
        ['Ceiling reached its limit (1.0): the data show no saturation, so the upper end of the curve is ' ...
         'unconstrained. Thickness predicted for DPM above the highest observed value (%.3f) is extrapolation.'], ...
        observed_dpm_range(2));
end
if fitted_floor_dpm <= 1e-3
    warning('fit_thickness_from_dpm:floorAtLimit', ...
        ['Floor reached its limit (0): the data show no noise floor, so the lower end of the curve is ' ...
         'unconstrained. Thickness predicted for DPM below the lowest observed value (%.3f) is extrapolation.'], ...
        observed_dpm_range(1));
end
if fitted_t50_mm < class_lower_mm(1) || fitted_t50_mm > class_upper_mm(end)
    warning('fit_thickness_from_dpm:t50Extrapolated', ...
        't50 (%.2f mm) lies outside the calibrated thickness range [%g, %g] mm.', fitted_t50_mm, class_lower_mm(1), class_upper_mm(end));
end

% --- Figure: box plot on a log thickness axis, with both fitted curves ---
if make_plot
    figure_fit = figure('Position', [100 100 900 680]);
    hold on;
    thickness_axis_limits_mm = [class_lower_mm(1) / 2.5, class_upper_mm(end) * 3];   % margins so no tick label sits on the edge
    % Isopach class edges as faint horizontal lines, for reference
    for edge_value_mm = isopach_edges_mm
        yline(edge_value_mm, ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
    end
    draw_boxes_on_log_axis(dpm_data, class_index_data, class_lower_mm, class_upper_mm, class_median_log10_thickness);
    plot(class_median_dpm, 10 .^ class_median_log10_thickness, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 7, ...
        'DisplayName', 'Class median (DPM, thickness): points of the median fit');
    % Curves only between floor and ceiling, where they are defined
    median_curve_dpm = linspace(median_fit_params(3) + dpm_clip_margin, median_fit_params(4) - dpm_clip_margin, 400)';
    pixel_curve_dpm = linspace(pixel_fit_params(3) + dpm_clip_margin, pixel_fit_params(4) - dpm_clip_margin, 400)';
    plot(pixel_curve_dpm, 10 .^ predict_log10_thickness(pixel_curve_dpm, pixel_fit_params, dpm_clip_margin), '--', ...
        'Color', [0.1 0.4 0.9], 'LineWidth', 1.8, 'DisplayName', sprintf('Fit to all pixels: t_{50}=%.1f mm, k=%.2f, floor=%.3f, ceiling=%.3f', ...
        10 ^ pixel_fit_params(2), pixel_fit_params(1), pixel_fit_params(3), pixel_fit_params(4)));
    plot(median_curve_dpm, 10 .^ predict_log10_thickness(median_curve_dpm, median_fit_params, dpm_clip_margin), 'r-', ...
        'LineWidth', 2.2, 'DisplayName', sprintf('Fit to class medians: t_{50}=%.1f mm, k=%.2f, floor=%.3f, ceiling=%.3f', ...
        10 ^ median_fit_params(2), median_fit_params(1), median_fit_params(3), median_fit_params(4)));
    hold off;
    set(gca, 'YScale', 'log');
    xlim([0 1]);
    ylim(thickness_axis_limits_mm);
    xlabel('DPM (probability of coherence change, dimensionless, 0-1)');
    ylabel('Ash thickness (mm, log scale)');
    title(sprintf('Inverse sigmoid fitted to the box-plot medians (carried forward: %s)', fit_target_label));
    % In-sample pixel scores of both fits in a box inside the axes (lower right)
    text(0.98, 0.04, {'In-sample, all pixels (optimistic):', ...
        sprintf('Medians fit: RMSE %.3f log10 mm, %.0f%% within \\times2', median_fit_scores.rmse_log10_mm, 100 * median_fit_scores.within_factor_2), ...
        sprintf('Pixels fit:  RMSE %.3f log10 mm, %.0f%% within \\times2', pixel_fit_scores.rmse_log10_mm, 100 * pixel_fit_scores.within_factor_2)}, ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
        'BackgroundColor', 'w', 'EdgeColor', [0.6 0.6 0.6], 'Margin', 4);
    legend('Location', 'southoutside', 'NumColumns', 1);
    grid on;
    box on;
    axtoolbar(gca, {});
    exportgraphics(figure_fit, fullfile(output_directory, [output_basename '_02_fit.png']), 'Resolution', 150);   % PNG at 150 dpi, cropped to the figure
    exportgraphics(figure_fit, fullfile(output_directory, [output_basename '_02_fit.pdf']), 'ContentType', 'vector');   % PDF page = figure size (print -dpdf shrinks it onto US letter)
end

% --- Out-of-bag (OOB) bootstrap: uncertainty of the curve AND hold-out test ---
% Each round: resample pixels WITH replacement within each class (same class sizes), refit
% both versions, then predict the pixels NOT drawn in that round (~37%, "out-of-bag").
% Those pixels played no part in that round's fit, so they are a genuine hold-out for it.
% NOTE: pixels are resampled independently; neighbouring DPM pixels are spatially correlated,
% so the intervals and OOB scores are optimistic for a new area.
fprintf('[4/5] Out-of-bag bootstrap (%d rounds, resampling within each class)...\n', number_of_bootstraps);
t0 = tic;
bootstrap_stream = RandStream('mt19937ar', 'Seed', random_seed + 2);   % private stream, caller's rng untouched
number_of_pixels = numel(dpm_data);
pixel_list_per_class = cell(number_of_classes, 1);
for class_number = 1:number_of_classes
    pixel_list_per_class{class_number} = find(class_index_data == class_number);
end
fit_names = {'median_fit', 'pixel_fit'};                 % order used in all bootstrap arrays below
full_data_params = {median_fit_params, pixel_fit_params};
bootstrap_params = {nan(number_of_bootstraps, 4), nan(number_of_bootstraps, 4)};   % [k, log10 t50, floor, ceiling]
round_scores = {nan(number_of_bootstraps, 3), nan(number_of_bootstraps, 3)};       % [RMSE log10 mm, R^2, within x2]
oob_prediction_sum = {zeros(number_of_pixels, 1), zeros(number_of_pixels, 1)};     % running sums (memory-light)
oob_count = {zeros(number_of_pixels, 1), zeros(number_of_pixels, 1)};   % per fit: rounds in which a pixel was OOB AND that fit succeeded
failed_round_count = [0, 0];
nonconverged_round_count = [0, 0];
% Pixels the carried-forward prediction will actually give a thickness for: DPM strictly inside
% the full-data fit's floor/ceiling. Only these feed the prediction interval, so it is calibrated
% on the same kind of pixel it is later applied to (predict_thickness_from_dpm returns NaN otherwise)
is_predictable = cell(1, 2);
for fit_number = 1:2
    is_predictable{fit_number} = dpm_data > full_data_params{fit_number}(3) + dpm_clip_margin & ...
                                 dpm_data < full_data_params{fit_number}(4) - dpm_clip_margin;
end
% Pooled OOB residuals (actual - predicted, log10 mm) kept as fine histograms per class,
% not as raw values, so memory stays small for any number of pixels
residual_bin_edges = -4:0.002:4;                                   % log10 mm; +/-4 = factor 10^4
number_of_residual_bins = numel(residual_bin_edges) - 1;
residual_histogram = {zeros(number_of_classes, number_of_residual_bins), zeros(number_of_classes, number_of_residual_bins)};
for bootstrap_number = 1:number_of_bootstraps
    % Stratified resample: draw n_c pixels with replacement from each class
    resample_list = zeros(number_of_pixels, 1);
    write_position = 0;
    for class_number = 1:number_of_classes
        class_pixels = pixel_list_per_class{class_number};
        drawn_pixels = class_pixels(randi(bootstrap_stream, numel(class_pixels), numel(class_pixels), 1));
        resample_list(write_position + (1:numel(drawn_pixels))) = drawn_pixels;
        write_position = write_position + numel(drawn_pixels);
    end
    is_out_of_bag = true(number_of_pixels, 1);
    is_out_of_bag(resample_list) = false;
    resampled_dpm = dpm_data(resample_list);
    resampled_log10_thickness = log10_thickness_data(resample_list);
    resampled_class = class_index_data(resample_list);

    for fit_number = 1:2
        start_floor_ceiling = full_data_params{fit_number}(3:4);   % start at the full-data fit (fast)
        if fit_number == 1   % median fit: 7 resampled class medians
            [round_params, round_exit_flag] = fit_inverse_sigmoid( ...
                accumarray(resampled_class, resampled_dpm, [number_of_classes 1], @median), ...
                accumarray(resampled_class, resampled_log10_thickness, [number_of_classes 1], @median), ...
                dpm_clip_margin, start_floor_ceiling);
        else                 % pixel fit: every resampled pixel
            [round_params, round_exit_flag] = fit_inverse_sigmoid(resampled_dpm, resampled_log10_thickness, dpm_clip_margin, start_floor_ceiling);
        end
        if any(~isfinite(round_params))
            failed_round_count(fit_number) = failed_round_count(fit_number) + 1;   % no valid fit this round
            continue;
        end
        if round_exit_flag ~= 1
            nonconverged_round_count(fit_number) = nonconverged_round_count(fit_number) + 1;   % kept, but counted
        end
        bootstrap_params{fit_number}(bootstrap_number, :) = round_params;
        % Hold-out pixels of this round that the final prediction WOULD predict (DPM strictly inside
        % the full-data floor/ceiling); saturated / below-floor pixels are reported separately
        oob_scored = is_out_of_bag & is_predictable{fit_number};
        oob_count{fit_number}(oob_scored) = oob_count{fit_number}(oob_scored) + 1;   % only rounds that contributed
        oob_prediction = predict_log10_thickness(dpm_data(oob_scored), round_params, dpm_clip_margin);
        this_round = score_log10_prediction(oob_prediction, log10_thickness_data(oob_scored));
        round_scores{fit_number}(bootstrap_number, :) = [this_round.rmse_log10_mm, this_round.r_squared, this_round.within_factor_2];
        oob_prediction_sum{fit_number}(oob_scored) = oob_prediction_sum{fit_number}(oob_scored) + oob_prediction;
        % Residuals for the prediction interval, from the same pixels
        oob_residual = log10_thickness_data(oob_scored) - oob_prediction;
        oob_residual = min(max(oob_residual, residual_bin_edges(1)), residual_bin_edges(end));   % clip into the histogram range
        residual_bin = discretize(oob_residual, residual_bin_edges);
        residual_histogram{fit_number} = residual_histogram{fit_number} + ...
            accumarray([class_index_data(oob_scored), residual_bin], 1, [number_of_classes, number_of_residual_bins]);
    end
end
fprintf('  Done in %.1fs\n', toc(t0));
carried_fit_number = 1 + strcmp(fit_target, 'pixels');   % 1 = median_fit, 2 = pixel_fit
for fit_number = 1:2
    failed_fraction = failed_round_count(fit_number) / number_of_bootstraps;
    if fit_number == carried_fit_number && failed_fraction > 0.2
        % Too many failures: the surviving rounds are a biased subset, so the intervals cannot be trusted
        error('fit_thickness_from_dpm:bootstrapFailed', ...
            ['%s (carried forward): %d of %d bootstrap rounds (%.0f%%) gave no valid fit. Stopping: the ' ...
             'relationship is too weak for this data size. Merge classes (edit isopach_edges_mm in your call) ' ...
             'or try FitTarget ''pixels'' (default_fit_target at %s).'], fit_names{fit_number}, ...
            failed_round_count(fit_number), number_of_bootstraps, 100 * failed_fraction, settings_location('default_fit_target'));
    elseif failed_round_count(fit_number) > 0
        warning('fit_thickness_from_dpm:bootstrapFailures', '%s: %d of %d bootstrap rounds gave no valid fit and were skipped.', ...
            fit_names{fit_number}, failed_round_count(fit_number), number_of_bootstraps);
    end
    if nonconverged_round_count(fit_number) > 0.05 * number_of_bootstraps
        warning('fit_thickness_from_dpm:bootstrapNotConverged', ...
            '%s: %d of %d bootstrap rounds hit the iteration limit (kept); their curves may be slightly off.', ...
            fit_names{fit_number}, nonconverged_round_count(fit_number), number_of_bootstraps);
    end
end
never_out_of_bag = oob_count{carried_fit_number} == 0;   % includes unpredictable pixels (never scored)
unpredictable_pixel = ~is_predictable{carried_fit_number};
unpredictable_fraction_per_class = accumarray(class_index_data, double(unpredictable_pixel), [number_of_classes 1], @mean);
if any(never_out_of_bag & ~unpredictable_pixel)
    fprintf('  %d predictable pixel(s) were never out-of-bag and are left out of the OOB scores\n', ...
        sum(never_out_of_bag & ~unpredictable_pixel));
end
scored_counts = oob_count{carried_fit_number}(~unpredictable_pixel);
fprintf('  Each scored pixel was out-of-bag in %g-%g rounds (median %g)\n', min(scored_counts), max(scored_counts), median(scored_counts));
fprintf('  Unpredictable pixels (DPM at/below floor or at/above ceiling; no thickness, not scored): %d of %d\n', ...
    sum(unpredictable_pixel), numel(unpredictable_pixel));
if any(unpredictable_pixel)
    for class_number = find(unpredictable_fraction_per_class > 0)'
        fprintf('    %-12s %.0f%% unpredictable\n', sprintf('%g-%g', class_lower_mm(class_number), class_upper_mm(class_number)), ...
            100 * unpredictable_fraction_per_class(class_number));
    end
end

% --- Summaries for both fits ---
dpm_band_grid = linspace(min(dpm_data), max(dpm_data), 300)';   % band only over the OBSERVED DPM range
summary = struct();
for fit_number = 1:2
    valid_rounds = all(isfinite(bootstrap_params{fit_number}), 2);
    params_matrix = bootstrap_params{fit_number}(valid_rounds, :);
    if size(params_matrix, 1) < 2   % only reachable for the non-carried fit (the carried one stops above)
        params_matrix = repmat(full_data_params{fit_number}, 2, 1);
        warning('fit_thickness_from_dpm:noBootstrap', '%s: fewer than 2 valid bootstrap rounds; its band and scores are placeholders.', fit_names{fit_number});
    end
    % Parameter 95% intervals (t50 interval taken on log10 t50, then converted to mm)
    parameter_ci_95 = quantile(params_matrix, [0.025 0.975], 1);   % 2-by-4
    % Confidence band of the curve: 2.5/97.5% of all bootstrap curves at each DPM (log10 mm)
    band_curves_log10_mm = zeros(numel(dpm_band_grid), size(params_matrix, 1));
    for round_number = 1:size(params_matrix, 1)
        band_curves_log10_mm(:, round_number) = predict_log10_thickness(dpm_band_grid, params_matrix(round_number, :), dpm_clip_margin);
    end
    band_log10_mm = quantile(band_curves_log10_mm, [0.025 0.975], 2);   % 300-by-2
    % OOB scores: (a) per round, (b) per pixel, from the average of its OOB predictions
    valid_scores = round_scores{fit_number}(all(isfinite(round_scores{fit_number}), 2), :);
    % Average OOB prediction per pixel = a BAGGED prediction (average of many fits): its error is
    % lower than that of the single fit you will use, so the per-round scores are the headline
    pixel_oob_log10_mm = oob_prediction_sum{fit_number} ./ oob_count{fit_number};
    has_oob = oob_count{fit_number} > 0;
    pixel_oob_log10_mm(~has_oob) = NaN;
    aggregated = score_log10_prediction(pixel_oob_log10_mm(has_oob), log10_thickness_data(has_oob));
    oob_residual_log10_mm = log10_thickness_data - pixel_oob_log10_mm;   % actual - predicted
    class_oob_rmse = accumarray(class_index_data(has_oob), oob_residual_log10_mm(has_oob) .^ 2, [number_of_classes 1], @mean) .^ 0.5;
    class_oob_bias = accumarray(class_index_data(has_oob), oob_residual_log10_mm(has_oob), [number_of_classes 1], @mean);
    summary.(fit_names{fit_number}) = struct( ...
        'params', params_matrix, ...
        'parameter_ci_95', parameter_ci_95, ...
        'band_dpm', dpm_band_grid, ...
        'band_log10_mm', band_log10_mm, ...
        'round_rmse_log10_mm', valid_scores(:, 1), ...
        'round_within_factor_2', valid_scores(:, 3), ...
        'oob_rmse_log10_mm', aggregated.rmse_log10_mm, ...
        'oob_r_squared', aggregated.r_squared, ...
        'oob_within_factor_2', aggregated.within_factor_2, ...
        'pixel_oob_log10_mm', pixel_oob_log10_mm, ...
        'class_oob_rmse_log10_mm', class_oob_rmse, ...
        'class_oob_bias_log10_mm', class_oob_bias, ...
        'failed_rounds', failed_round_count(fit_number));
end

% --- Report ---
fprintf('  95%% bootstrap intervals of the parameters:\n');
fprintf('  %-28s %22s %22s\n', '', 'Medians fit', 'Pixels fit');
fprintf('  %-28s %22s %22s\n', 't50 (mm)', ...
    sprintf('%.1f [%.1f, %.1f]', 10 ^ median_fit_params(2), 10 .^ summary.median_fit.parameter_ci_95(:, 2)), ...
    sprintf('%.1f [%.1f, %.1f]', 10 ^ pixel_fit_params(2), 10 .^ summary.pixel_fit.parameter_ci_95(:, 2)));
parameter_names = {'steepness k (per log10 mm)', '', 'floor (DPM)', 'ceiling (DPM)'};
for parameter_number = [1 3 4]
    fprintf('  %-28s %22s %22s\n', parameter_names{parameter_number}, ...
        sprintf('%.3f [%.3f, %.3f]', median_fit_params(parameter_number), summary.median_fit.parameter_ci_95(:, parameter_number)), ...
        sprintf('%.3f [%.3f, %.3f]', pixel_fit_params(parameter_number), summary.pixel_fit.parameter_ci_95(:, parameter_number)));
end
fprintf('  OUT-OF-BAG (hold-out) scores:\n');
fprintf('  %-40s %14s %14s\n', '', 'Medians fit', 'Pixels fit');
fprintf('  Single fit (per bootstrap round; HEADLINE - this is the fit you will use):\n');
fprintf('  %-40s %14s %14s\n', 'RMSE, median [95%] (log10 mm)', ...
    sprintf('%.3f [%.3f,%.3f]', median(summary.median_fit.round_rmse_log10_mm), quantile(summary.median_fit.round_rmse_log10_mm, [0.025 0.975])), ...
    sprintf('%.3f [%.3f,%.3f]', median(summary.pixel_fit.round_rmse_log10_mm), quantile(summary.pixel_fit.round_rmse_log10_mm, [0.025 0.975])));
fprintf('  %-40s %13.0f%% %13.0f%%\n', 'Within x2, median', 100 * median(summary.median_fit.round_within_factor_2), ...
    100 * median(summary.pixel_fit.round_within_factor_2));
fprintf('  Bagged (each pixel''s OOB predictions averaged over rounds; slightly optimistic):\n');
fprintf('  %-40s %14.3f %14.3f\n', 'RMSE (log10 mm)', summary.median_fit.oob_rmse_log10_mm, summary.pixel_fit.oob_rmse_log10_mm);
fprintf('  %-40s %13.0f%% %13.0f%%\n', 'Within x2', 100 * summary.median_fit.oob_within_factor_2, 100 * summary.pixel_fit.oob_within_factor_2);
fprintf('  %-40s %14.3f %14.3f\n', 'R^2 in log10 thickness', summary.median_fit.oob_r_squared, summary.pixel_fit.oob_r_squared);
fprintf('  Per class OOB RMSE / bias (log10 mm; bias > 0 = fit predicts too THIN):\n');
fprintf('  %-14s %22s %22s\n', 'Class (mm)', 'Medians fit', 'Pixels fit');
for class_number = 1:number_of_classes
    fprintf('  %-14s %22s %22s\n', sprintf('%g-%g', class_lower_mm(class_number), class_upper_mm(class_number)), ...
        sprintf('%.3f / %+.3f', summary.median_fit.class_oob_rmse_log10_mm(class_number), summary.median_fit.class_oob_bias_log10_mm(class_number)), ...
        sprintf('%.3f / %+.3f', summary.pixel_fit.class_oob_rmse_log10_mm(class_number), summary.pixel_fit.class_oob_bias_log10_mm(class_number)));
end
fprintf('  (pixels resampled as independent: real-scene scores will be worse because neighbouring pixels are correlated)\n');
carried_summary = summary.(fit_names{1 + strcmp(fit_target, 'pixels')});

% --- Figure: bootstrap diagnostics (2-by-2) ---
if make_plot
    figure_bootstrap = figure('Position', [100 100 1250 950]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    fit_colours = {[0.85 0.1 0.1], [0.1 0.4 0.9]};   % red = medians fit, blue = pixels fit
    fit_labels = {'Medians fit', 'Pixels fit'};

    % (a) Both curves with their 95% bootstrap confidence bands
    nexttile;
    hold on;
    for edge_value_mm = isopach_edges_mm   % isopach class edges, faint, for reference
        yline(edge_value_mm, ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
    end
    draw_boxes_on_log_axis(dpm_data, class_index_data, class_lower_mm, class_upper_mm, class_median_log10_thickness);
    for fit_number = [2 1]
        this_summary = summary.(fit_names{fit_number});
        fill([dpm_band_grid; flipud(dpm_band_grid)], 10 .^ [this_summary.band_log10_mm(:, 1); flipud(this_summary.band_log10_mm(:, 2))], ...
            fit_colours{fit_number}, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', [fit_labels{fit_number} ' 95% CI']);
        plot(dpm_band_grid, 10 .^ predict_log10_thickness(dpm_band_grid, full_data_params{fit_number}, dpm_clip_margin), '-', ...
            'Color', fit_colours{fit_number}, 'LineWidth', 2, 'DisplayName', fit_labels{fit_number});
    end
    plot(class_median_dpm, 10 .^ class_median_log10_thickness, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, 'DisplayName', 'Class medians');
    hold off;
    set(gca, 'YScale', 'log');
    xlim([0 1]);
    ylim([class_lower_mm(1) / 2.5, class_upper_mm(end) * 3]);
    yticks(10 .^ (floor(log10(class_lower_mm(1) / 2.5)):ceil(log10(class_upper_mm(end) * 3))));   % label every decade
    xlabel('DPM (dimensionless, 0-1)');
    ylabel('Ash thickness (mm, log scale)');
    title({'(a) Best fits with 95% bootstrap band', sprintf('(%d rounds, observed DPM range only)', number_of_bootstraps)});
    legend('Location', 'southeast', 'FontSize', 8);   % lower right is empty (thick ash never has low DPM)
    grid on; box on;

    % (b) OOB predicted vs actual thickness, carried-forward fit
    nexttile;
    has_oob = ~never_out_of_bag;
    loglog(thickness_data_mm(has_oob), 10 .^ carried_summary.pixel_oob_log10_mm(has_oob), 'o', 'Color', [0.3 0.3 0.3], ...
        'MarkerFaceColor', [0.6 0.6 0.6], 'MarkerSize', 4, 'DisplayName', 'Pixels (bagged OOB prediction)');
    hold on;
    one_to_one_mm = [class_lower_mm(1) / 2.5, class_upper_mm(end) * 3];
    plot(one_to_one_mm, one_to_one_mm, 'k-', 'LineWidth', 1.5, 'DisplayName', '1:1');
    plot(one_to_one_mm, one_to_one_mm * 2, 'k--', 'DisplayName', '\times2 / \div2');
    plot(one_to_one_mm, one_to_one_mm / 2, 'k--', 'HandleVisibility', 'off');
    hold off;
    xlim(one_to_one_mm);
    ylim(one_to_one_mm);
    axis square;
    xlabel('Actual thickness (mm)');
    ylabel('Predicted thickness, out-of-bag (mm)');
    title({sprintf('(b) Out-of-bag hold-out test, %s', lower(fit_labels{1 + strcmp(fit_target, 'pixels')})), ...
        sprintf('bagged: RMSE %.3f log10 mm, %.0f%% within \\times2', carried_summary.oob_rmse_log10_mm, 100 * carried_summary.oob_within_factor_2)});
    legend('Location', 'northwest', 'FontSize', 8);
    grid on; box on;

    % (c) Per-round OOB RMSE for both fits
    nexttile;
    hold on;
    all_round_rmse = [summary.median_fit.round_rmse_log10_mm; summary.pixel_fit.round_rmse_log10_mm];
    bin_edges = linspace(min(all_round_rmse), max(all_round_rmse), 35);
    for fit_number = 1:2
        histogram(summary.(fit_names{fit_number}).round_rmse_log10_mm, bin_edges, 'FaceColor', fit_colours{fit_number}, ...
            'FaceAlpha', 0.45, 'EdgeColor', 'none', 'DisplayName', sprintf('%s (median %.3f)', fit_labels{fit_number}, ...
            median(summary.(fit_names{fit_number}).round_rmse_log10_mm)));
    end
    hold off;
    xlabel('Out-of-bag RMSE per round (log10 mm)');
    ylabel('Number of bootstrap rounds');
    title('(c) Hold-out error across bootstrap rounds');
    legend('Location', 'northeast', 'FontSize', 8);
    grid on; box on;

    % (d) Per-class OOB bias for both fits (mean residual = actual - predicted)
    nexttile;
    class_labels_short = arrayfun(@(lower_mm, upper_mm) sprintf('%g-%g', lower_mm, upper_mm), class_lower_mm, class_upper_mm, 'UniformOutput', false);
    bar_handles = bar(1:number_of_classes, [summary.median_fit.class_oob_bias_log10_mm, summary.pixel_fit.class_oob_bias_log10_mm]);
    bar_handles(1).FaceColor = fit_colours{1};
    bar_handles(2).FaceColor = fit_colours{2};
    bar_handles(1).DisplayName = fit_labels{1};
    bar_handles(2).DisplayName = fit_labels{2};
    yline(0, 'k-', 'HandleVisibility', 'off');
    yline([-1 1] * log10(2), 'k:', 'HandleVisibility', 'off');
    xticks(1:number_of_classes);
    xticklabels(class_labels_short);
    xtickangle(30);
    xlabel('Ash thickness class (mm)');
    ylabel('Mean OOB residual, actual - predicted (log10 mm)');
    title({'(d) Out-of-bag bias per class', '(> 0: predicts too thin; dotted = \times2)'});
    legend('Location', 'best', 'FontSize', 8);
    grid on; box on;

    exportgraphics(figure_bootstrap, fullfile(output_directory, [output_basename '_03_bootstrap.png']), 'Resolution', 150);   % PNG at 150 dpi, cropped to the figure
    exportgraphics(figure_bootstrap, fullfile(output_directory, [output_basename '_03_bootstrap.pdf']), 'ContentType', 'vector');   % PDF page = figure size (print -dpdf shrinks it onto US letter)
end

% --- Prediction interval for a single pixel, from the pooled out-of-bag residuals ---
% PI = curve + [lower, upper] quantile of (actual - predicted) log10 thickness, where every
% residual comes from a bootstrap fit that did NOT see that pixel. It therefore includes both
% the curve uncertainty and the pixel-to-pixel scatter. Constant width in log10 thickness
% (= a constant factor): with ~50 pixels per class, per-class 2.5/97.5% quantiles would rest
% on 1-2 pixels each, so the per-class coverage below is the check on this assumption.
fprintf('[5/5] Plot 2: prediction interval from pooled out-of-bag residuals...\n');
pooled_counts = residual_histogram{carried_fit_number};          % classes-by-bins
total_counts = sum(pooled_counts, 1);
cumulative_fraction = cumsum(total_counts) / sum(total_counts);
tail_fraction = (1 - prediction_coverage) / 2;
% Lower limit: left edge of the bin where the CDF first reaches the lower tail; upper: right edge
% of the bin where it first reaches the upper tail (conservative by at most one bin width)
pi_low_log10_mm = residual_bin_edges(find(cumulative_fraction >= tail_fraction, 1));
pi_high_log10_mm = residual_bin_edges(find(cumulative_fraction >= 1 - tail_fraction, 1) + 1);
% Coverage of this interval in each class (fraction of that class's OOB residuals inside it)
bin_centres = (residual_bin_edges(1:end-1) + residual_bin_edges(2:end)) / 2;
inside_bins = bin_centres >= pi_low_log10_mm & bin_centres <= pi_high_log10_mm;
class_pi_coverage = sum(pooled_counts(:, inside_bins), 2) ./ sum(pooled_counts, 2);
overall_pi_coverage = sum(total_counts(inside_bins)) / sum(total_counts);
fprintf('  %.0f%% prediction interval: x%.2f to x%.2f around the curve (%.3f to %+.3f log10 mm; width x%.1f)\n', ...
    100 * prediction_coverage, 10 ^ pi_low_log10_mm, 10 ^ pi_high_log10_mm, pi_low_log10_mm, pi_high_log10_mm, ...
    10 ^ (pi_high_log10_mm - pi_low_log10_mm));
fprintf('  Coverage of the out-of-bag residuals: overall %.1f%% (by construction), per class:\n', 100 * overall_pi_coverage);
for class_number = 1:number_of_classes
    fprintf('    %-12s %5.1f%%\n', sprintf('%g-%g', class_lower_mm(class_number), class_upper_mm(class_number)), ...
        100 * class_pi_coverage(class_number));
end
low_coverage_classes = find(class_pi_coverage < prediction_coverage - 0.05);
if ~isempty(low_coverage_classes)
    warning('fit_thickness_from_dpm:piUndercoverage', ...
        ['The prediction interval covers < %.0f%% in class(es) %s: scatter there is larger than average, ' ...
         'so single-pixel predictions in that range are less certain than the PI says.'], ...
        100 * (prediction_coverage - 0.05), strjoin(arrayfun(@(class_number) sprintf('%g-%g mm', ...
        class_lower_mm(class_number), class_upper_mm(class_number)), low_coverage_classes', 'UniformOutput', false), ', '));
end

% --- Plot 2: final calibration curve, DPM -> ash thickness ---
if make_plot
    figure_plot2 = figure('Position', [100 100 950 760]);
    hold on;
    for edge_value_mm = isopach_edges_mm   % isopach class edges, faint, for reference
        yline(edge_value_mm, ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
    end
    draw_boxes_on_log_axis(dpm_data, class_index_data, class_lower_mm, class_upper_mm, class_median_log10_thickness);
    curve_log10_mm = predict_log10_thickness(dpm_band_grid, fitted_params, dpm_clip_margin);   % observed DPM range
    fill([dpm_band_grid; flipud(dpm_band_grid)], 10 .^ [curve_log10_mm + pi_low_log10_mm; flipud(curve_log10_mm + pi_high_log10_mm)], ...
        [0.2 0.45 0.9], 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
        'DisplayName', sprintf('%.0f%% prediction interval, single pixel (\\times%.2f to \\times%.2f)', ...
        100 * prediction_coverage, 10 ^ pi_low_log10_mm, 10 ^ pi_high_log10_mm));
    fill([dpm_band_grid; flipud(dpm_band_grid)], 10 .^ [carried_summary.band_log10_mm(:, 1); flipud(carried_summary.band_log10_mm(:, 2))], ...
        [0.85 0.1 0.1], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', '95% confidence interval of the curve (bootstrap)');
    % Extrapolation beyond the observed DPM range, up to the ceiling: dashed
    extrapolation_dpm = linspace(max(dpm_data), fitted_ceiling_dpm - dpm_clip_margin, 100)';
    if numel(extrapolation_dpm) > 1 && extrapolation_dpm(end) > extrapolation_dpm(1)
        plot(extrapolation_dpm, 10 .^ predict_log10_thickness(extrapolation_dpm, fitted_params, dpm_clip_margin), 'r--', ...
            'LineWidth', 1.2, 'DisplayName', 'Extrapolation (beyond observed DPM)');
    end
    plot(dpm_band_grid, 10 .^ curve_log10_mm, 'r-', 'LineWidth', 2.4, 'DisplayName', ...
        sprintf('Best fit (%s): t_{50}=%.1f mm, k=%.2f, floor=%.3f, ceiling=%.3f', fit_target_label, ...
        fitted_t50_mm, fitted_steepness, fitted_floor_dpm, fitted_ceiling_dpm));
    plot(class_median_dpm, 10 .^ class_median_log10_thickness, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
        'DisplayName', 'Class medians');
    hold off;
    set(gca, 'YScale', 'log');
    xlim([0 1]);
    ylim([class_lower_mm(1) / 2.5, class_upper_mm(end) * 3]);
    yticks(10 .^ (floor(log10(class_lower_mm(1) / 2.5)):ceil(log10(class_upper_mm(end) * 3))));
    xlabel('DPM (probability of coherence change, dimensionless, 0-1)');
    ylabel('Ash thickness (mm, log scale)');
    title({'Ash thickness from DPM: inverse sigmoid', ...
        sprintf('log_{10} t = log_{10} %.1f + (1/%.2f) ln[(DPM - %.3f)/(%.3f - DPM)]', ...
        fitted_t50_mm, fitted_steepness, fitted_floor_dpm, fitted_ceiling_dpm)});
    % Hold-out scores in a box inside the axes (lower right, which has no data)
    headline_rmse = median(carried_summary.round_rmse_log10_mm);
    headline_rmse_range = quantile(carried_summary.round_rmse_log10_mm, [0.025 0.975]);
    text(0.98, 0.04, {sprintf('Out-of-bag hold-out, %d bootstrap fits:', number_of_bootstraps), ...
        sprintf('RMSE = %.3f log_{10} mm (typical error \\times%.2f)', headline_rmse, 10 ^ headline_rmse), ...
        sprintf('   95%% of fits: %.3f to %.3f log_{10} mm', headline_rmse_range(1), headline_rmse_range(2)), ...
        sprintf('%.0f%% of pixels within \\times2', 100 * median(carried_summary.round_within_factor_2)), ...
        sprintf('n = %d pixels, %d classes; %.0f%% unpredictable', numel(dpm_data), number_of_classes, 100 * mean(unpredictable_pixel))}, ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
        'BackgroundColor', 'w', 'EdgeColor', [0.6 0.6 0.6], 'Margin', 4, 'FontSize', 9);
    legend('Location', 'southoutside', 'NumColumns', 1, 'FontSize', 9);
    grid on;
    box on;
    axtoolbar(gca, {});
    exportgraphics(figure_plot2, fullfile(output_directory, [output_basename '_04_plot2.png']), 'Resolution', 150);   % PNG at 150 dpi, cropped to the figure
    exportgraphics(figure_plot2, fullfile(output_directory, [output_basename '_04_plot2.pdf']), 'ContentType', 'vector');   % PDF page = figure size (print -dpdf shrinks it onto US letter)
end

% --- Collect outputs ---
fit_results = struct( ...
    'dpm', dpm_data, ...
    'thickness_mm', thickness_data_mm, ...
    'class_index', class_index_data, ...
    'class_table', table(class_lower_mm, class_upper_mm, pixel_count_per_class, class_median_dpm, ...
        10 .^ class_median_log10_thickness, ...
        saturated_count_per_class, unpredictable_fraction_per_class, ...
        'VariableNames', {'lower_mm', 'upper_mm', 'pixel_count', 'median_dpm', 'median_thickness_mm', ...
        'saturated_count', 'unpredictable_fraction'}), ...
    'fit_target', fit_target, ...
    'steepness_per_log10_mm', fitted_steepness, ...
    't50_mm', fitted_t50_mm, ...
    'floor_dpm', fitted_floor_dpm, ...
    'ceiling_dpm', fitted_ceiling_dpm, ...
    'params', fitted_params, ...
    'dpm_clip_margin', dpm_clip_margin, ...
    'exit_flag', exit_flag, ...
    'observed_dpm_range', observed_dpm_range, ...
    'calibrated_thickness_range_mm', [class_lower_mm(1), class_upper_mm(end)], ...
    'in_sample', in_sample_scores, ...
    'median_fit', struct('params', median_fit_params, 'in_sample', median_fit_scores, ...
        'residual_at_medians_log10_mm', median_fit_residual_at_medians), ...
    'pixel_fit', struct('params', pixel_fit_params, 'in_sample', pixel_fit_scores), ...
    'bootstrap', summary, ...
    'parameter_ci_95', carried_summary.parameter_ci_95, ...
    'oob', struct( ...
        'rmse_log10_mm', median(carried_summary.round_rmse_log10_mm), ...            % single fit, median over rounds (headline)
        'rmse_log10_mm_95', quantile(carried_summary.round_rmse_log10_mm, [0.025 0.975]), ...
        'within_factor_2', median(carried_summary.round_within_factor_2), ...
        'bagged_rmse_log10_mm', carried_summary.oob_rmse_log10_mm, ...              % averaged OOB predictions (optimistic)
        'bagged_within_factor_2', carried_summary.oob_within_factor_2, ...
        'bagged_r_squared', carried_summary.oob_r_squared), ...
    'prediction_interval', struct('coverage', prediction_coverage, ...
        'log10_mm', [pi_low_log10_mm, pi_high_log10_mm], ...
        'factor', 10 .^ [pi_low_log10_mm, pi_high_log10_mm], ...
        'class_coverage', class_pi_coverage));

fprintf('Done. Output saved to %s\n', output_directory);
end

% --- Local functions ---
function draw_boxes_on_log_axis(dpm_data, class_index_data, class_lower_mm, class_upper_mm, class_median_log10_thickness)
% DRAW_BOXES_ON_LOG_AXIS  Horizontal box-and-whisker of DPM per class on the CURRENT axes,
%   positioned on a real (log) thickness axis, which boxchart cannot do.
%   dpm_data                     : DPM of every pixel, dimensionless 0-1
%   class_index_data             : class number of every pixel (1..number of classes)
%   class_lower_mm/upper_mm      : class edges (mm)
%   class_median_log10_thickness : median log10 thickness per class (log10 mm), box centre
%   Box = interquartile range, thick line = median, whiskers = Tukey 1.5 IQR, circles = outliers.
%   Call with "hold on"; the y axis should be set to log scale afterwards.
    number_of_classes = numel(class_lower_mm);
    % Boxes drawn by hand so they can sit on a log thickness axis: each box is centred on the
    % class median thickness, its height is 60% of the class's log10 thickness span
    for class_number = 1:number_of_classes
        class_dpm = dpm_data(class_index_data == class_number);
        quartile_dpm = quantile(class_dpm, [0.25 0.5 0.75]);   % base-MATLAB quantile
        interquartile_range_dpm = quartile_dpm(3) - quartile_dpm(1);
        % Tukey whiskers: furthest pixel within 1.5 IQR of the box; beyond that = outlier
        inside_fence = class_dpm >= quartile_dpm(1) - 1.5 * interquartile_range_dpm & ...
                       class_dpm <= quartile_dpm(3) + 1.5 * interquartile_range_dpm;
        whisker_low_dpm = min(class_dpm(inside_fence));
        whisker_high_dpm = max(class_dpm(inside_fence));
        centre_log10_mm = class_median_log10_thickness(class_number);
        half_height_log10_mm = 0.3 * (log10(class_upper_mm(class_number)) - log10(class_lower_mm(class_number)));
        box_bottom_mm = 10 ^ (centre_log10_mm - half_height_log10_mm);
        box_top_mm = 10 ^ (centre_log10_mm + half_height_log10_mm);
        centre_mm = 10 ^ centre_log10_mm;
        show_in_legend = 'off';
        if class_number == 1
            show_in_legend = 'on';
        end
        patch(quartile_dpm([1 3 3 1]), [box_bottom_mm box_bottom_mm box_top_mm box_top_mm], [0.85 0.85 0.85], ...
            'EdgeColor', [0.4 0.4 0.4], 'DisplayName', 'DPM per class (box = IQR, whiskers = 1.5 IQR)', ...
            'HandleVisibility', show_in_legend);
        plot([whisker_low_dpm quartile_dpm(1)], [centre_mm centre_mm], '-', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
        plot([quartile_dpm(3) whisker_high_dpm], [centre_mm centre_mm], '-', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
        plot([whisker_low_dpm whisker_low_dpm], [box_bottom_mm ^ 0.75 * box_top_mm ^ 0.25, box_bottom_mm ^ 0.25 * box_top_mm ^ 0.75], ...
            '-', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');   % whisker caps, half the box height (log-centred)
        plot([whisker_high_dpm whisker_high_dpm], [box_bottom_mm ^ 0.75 * box_top_mm ^ 0.25, box_bottom_mm ^ 0.25 * box_top_mm ^ 0.75], ...
            '-', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
        plot([quartile_dpm(2) quartile_dpm(2)], [box_bottom_mm box_top_mm], '-', 'Color', [0.2 0.2 0.2], ...
            'LineWidth', 1.5, 'HandleVisibility', 'off');   % median line
        outlier_dpm = class_dpm(~inside_fence);
        plot(outlier_dpm, repmat(centre_mm, size(outlier_dpm)), 'o', 'Color', [0.4 0.4 0.4], 'MarkerSize', 4, ...
            'HandleVisibility', 'off');
    end
end

function log10_thickness = predict_log10_thickness(dpm_values, params, dpm_clip_margin)
% PREDICT_LOG10_THICKNESS  Inverse sigmoid: log10 ash thickness (log10 mm) from DPM.
%   dpm_values      : DPM, dimensionless 0-1 (any shape)
%   params          : [k (1/log10 mm), log10 t50 (log10 mm), floor (DPM), ceiling (DPM)]
%   dpm_clip_margin : DPM at/beyond floor or ceiling is clipped this far inside them,
%                     so the logit stays finite (DPM units)
%   Returns log10 thickness, same shape as dpm_values.
    steepness = params(1);
    log10_t50 = params(2);
    floor_dpm = params(3);
    ceiling_dpm = params(4);
    clipped_dpm = min(max(dpm_values, floor_dpm + dpm_clip_margin), ceiling_dpm - dpm_clip_margin);
    log10_thickness = log10_t50 + (1 / steepness) .* log((clipped_dpm - floor_dpm) ./ (ceiling_dpm - clipped_dpm));
end

function scores = score_log10_prediction(predicted_log10_mm, actual_log10_mm)
% SCORE_LOG10_PREDICTION  Error of predicted vs actual log10 thickness (both log10 mm).
%   Returns struct: rmse_log10_mm, r_squared (in log10 thickness), within_factor_2 (fraction 0-1).
    residual_log10_mm = actual_log10_mm - predicted_log10_mm;
    scores = struct( ...
        'rmse_log10_mm', sqrt(mean(residual_log10_mm .^ 2)), ...
        'r_squared', 1 - sum(residual_log10_mm .^ 2) / sum((actual_log10_mm - mean(actual_log10_mm)) .^ 2), ...
        'within_factor_2', mean(abs(residual_log10_mm) <= log10(2)));
end

function [fitted_params, exit_flag] = fit_inverse_sigmoid(dpm_values, log10_thickness_values, dpm_clip_margin, start_floor_ceiling)
% FIT_INVERSE_SIGMOID  Least-squares fit of log10 thickness = a + b * ln((DPM-floor)/(ceiling-DPM)).
%   For fixed floor/ceiling the model is linear in (a, b), so a = log10 t50 and b = 1/k are
%   solved exactly; fminsearch (base MATLAB) searches only over [floor, ceiling].
%   dpm_values             : DPM, dimensionless 0-1 (column vector)
%   log10_thickness_values : log10 ash thickness, log10 mm (column vector)
%   dpm_clip_margin        : see predict_log10_thickness (DPM units)
%   start_floor_ceiling    : [] = try a grid of starting points (full-data fit);
%                            [floor ceiling] = start there only (fast, for bootstrap rounds)
%   Returns fitted_params = [k, log10 t50, floor, ceiling] and fminsearch's exit flag.
    search_options = optimset('TolX', 1e-7, 'TolFun', 1e-10, 'MaxIter', 4000, 'MaxFunEvals', 8000);
    cost_function = @(floor_ceiling) profile_cost(floor_ceiling, dpm_values, log10_thickness_values, dpm_clip_margin);
    if isempty(start_floor_ceiling)
        % Several starts: floor below the lowest DPM, ceiling above the highest, at three distances each
        lowest_dpm = min(dpm_values);
        highest_dpm = max(dpm_values);
        floor_start_list = max(0, lowest_dpm - [0.005 0.02 0.05]);
        ceiling_start_list = min(1, highest_dpm + [0.005 0.02 0.05]);
        [floor_grid, ceiling_grid] = ndgrid(floor_start_list, ceiling_start_list);
        start_list = [floor_grid(:), ceiling_grid(:)];
    else
        start_list = start_floor_ceiling(:)';
    end
    best_cost = Inf;
    best_floor_ceiling = start_list(1, :);
    exit_flag = 0;
    for start_number = 1:size(start_list, 1)
        [floor_ceiling, cost, this_exit_flag] = fminsearch(cost_function, start_list(start_number, :), search_options);
        if cost < best_cost
            best_cost = cost;
            best_floor_ceiling = floor_ceiling;
            exit_flag = this_exit_flag;
        end
    end
    [~, intercept, slope] = profile_cost(best_floor_ceiling, dpm_values, log10_thickness_values, dpm_clip_margin);
    fitted_params = [1 / slope, intercept, best_floor_ceiling(1), best_floor_ceiling(2)];
end

function [cost, intercept, slope] = profile_cost(floor_ceiling, dpm_values, log10_thickness_values, dpm_clip_margin)
% PROFILE_COST  Sum of squared log10-thickness residuals for a given [floor, ceiling],
%   with the intercept (log10 t50) and slope (1/k) solved by linear least squares.
%   Invalid floor/ceiling, or a non-positive slope (thickness falling with DPM), get a large cost.
    floor_dpm = floor_ceiling(1);
    ceiling_dpm = floor_ceiling(2);
    intercept = NaN;
    slope = NaN;
    if floor_dpm < 0 || ceiling_dpm > 1 || ceiling_dpm - floor_dpm <= 4 * dpm_clip_margin
        cost = 1e10;
        return;
    end
    clipped_dpm = min(max(dpm_values, floor_dpm + dpm_clip_margin), ceiling_dpm - dpm_clip_margin);
    logit_term = log((clipped_dpm - floor_dpm) ./ (ceiling_dpm - clipped_dpm));
    coefficients = [ones(size(logit_term)), logit_term] \ log10_thickness_values;   % [intercept; slope]
    intercept = coefficients(1);
    slope = coefficients(2);
    if ~(slope > 0)
        % Return NaN, not the negative slope: if NO floor/ceiling gives a positive slope, the caller
        % must see "no valid fit" (NaN) instead of a finite curve with thickness falling as DPM rises
        cost = 1e10;
        intercept = NaN;
        slope = NaN;
        return;
    end
    cost = sum((log10_thickness_values - intercept - slope .* logit_term) .^ 2);
end

function location_text = settings_location(variable_name)
% SETTINGS_LOCATION  'file.m, line N' of a variable in the DEFAULT SETTINGS block,
%   so error/warning messages point to the exact line to edit.
%   variable_name : name of the default_* variable (char)
%   Returns char. In the MATLAB desktop the text is a clickable link that opens the line.
    this_file = [mfilename('fullpath') '.m'];
    line_number = 0;
    try
        file_lines = readlines(this_file);
        line_number = find(startsWith(strtrim(file_lines), [variable_name ' ']), 1);
    catch read_error
        % Fall back to a plain pointer if the file cannot be read; never hide the original error
        fprintf('  (could not read %s to find the settings line: %s)\n', this_file, read_error.message);
    end
    [~, file_name, file_extension] = fileparts(this_file);
    if isempty(line_number) || line_number == 0
        location_text = sprintf('the DEFAULT SETTINGS block of %s%s', file_name, file_extension);
    elseif usejava('desktop')
        location_text = sprintf('<a href="matlab: opentoline(''%s'', %d)">%s%s, line %d</a>', ...
            this_file, line_number, file_name, file_extension, line_number);
    else
        location_text = sprintf('%s%s, line %d', file_name, file_extension, line_number);
    end
end
