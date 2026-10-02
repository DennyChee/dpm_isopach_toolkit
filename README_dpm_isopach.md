# DPM → ash thickness toolkit

Calibrates a function that predicts **ash thickness from a Damage Proxy Map (DPM) pixel value**
(DPM = probability of coherence change, 0–1), using pixels where the thickness is known from isopachs.

The function is an **inverse sigmoid fitted directly in log10 thickness**:

```
log10 t = log10 t50 + (1/k) · ln[ (DPM − floor) / (ceiling − DPM) ]        (t in mm)
```

It is fitted with an **out-of-bag bootstrap**, which gives the curve's uncertainty and an honest
hold-out test without setting aside a fixed test set (important when data are limited, ~50 pixels per class).

| File | Purpose |
|---|---|
| `run_example.m` | **Start here.** Step-by-step walkthrough: input → fit → reading the result → predicting values → applying to a DPM map. Change only its EDIT HERE block for your own data |
| `fit_thickness_from_dpm.m` | Main function: N×2 table → box plot, fit, bootstrap, hold-out test, plot 2 |
| `predict_thickness_from_dpm.m` | DPM value(s) or a DPM map → thickness with confidence and prediction intervals |
| `preprocess_dpm_isopach.py` | DPM GeoTIFF + thickness GeoTIFF → the 2-column CSV |
| `make_example_dpm_csv.m` | Synthetic example with known truth. **Run this first**: it checks the fit against the truth and on independent test pixels |
| `old/` | The previous toolkit (forward fit, then invert), kept for reference |

Requirements: MATLAB R2022a or newer (base MATLAB; older releases need the Statistics and Machine Learning Toolbox for `quantile`); Python with `rasterio` and `numpy` (e.g. the `isce2` env).

## 1. Input

An **N-by-2 matrix, one row per pixel**, plus the isopach class edges:

| Column | Meaning |
|---|---|
| 1 | DPM value of the pixel, dimensionless, **scaled 0 to 1** |
| 2 | ash thickness at that pixel, in **mm** |

```matlab
edges_mm = [0.1 1 5 10 50 100 200 500];                     % 7 classes, increasing, all > 0
results  = fit_thickness_from_dpm(data_matrix, edges_mm);   % data_matrix is N-by-2
results  = fit_thickness_from_dpm('dpm_thickness.csv', edges_mm);   % or a CSV (header row allowed)
```

Rows with NaN and pixels outside [first edge, last edge] are dropped (the last edge itself is included). The edges define the box-plot classes,
the median fit (one point per class), and the stratified bootstrap.

## 2. What the function does

| Step | What | Figure (`<basename>_…`) |
|---|---|---|
| 1 | Load, validate, assign pixels to classes; **stop if any class is too small**, or if the medians fit has < 5 classes | `01_boxplot`: **plot 1**, DPM per thickness class |
| 2 | Fit the inverse sigmoid two ways: to the **class medians** (default) and to **all pixels** | `02_fit`: both fits on the box plot, log thickness axis |
| 3 | **Out-of-bag bootstrap** (1000 rounds): resample pixels within each class, refit, predict the ~37% of pixels left out | `03_bootstrap`: confidence bands, hold-out predicted vs actual, error per round, bias per class |
| 4 | Single-pixel **prediction interval** from the pooled out-of-bag residuals | `04_plot2`: **plot 2**, final calibration curve with CI, PI and hold-out scores |

### Medians fit vs pixels fit (`'FitTarget'`)

- **`'class_medians'` (default)**: one point per class (median DPM, median log10 thickness), the medians of the box plot.
  Gives every class equal weight and is much less pulled toward the middle of the data; check the printed per-class
  bias and figure `03_bootstrap` (d). With only one point per class it is less stable when the ceiling is weakly
  constrained (watch the confidence band above the top class). Needs at least 5 classes (4 parameters).
- **`'pixels'`**: every pixel. Lower and more stable typical error, but pulled toward the middle of the data:
  in the synthetic example it under-predicts the thickest class by ≈ ×1.5.

### Saturated DPM and unpredictable pixels

DPM at or near 1 only says "a lot of change", not how much: 30 cm and 3 m of ash can both read 1.
Saturated pixels stay in the fit (a class median is unaffected while fewer than half of its pixels are saturated),
the function warns or stops per class (settings above), and the class table reports `saturated_count`.
Pixels with DPM at/below the floor or at/above the ceiling get no thickness at prediction time, so they are
**left out of all scores** and reported instead as `unpredictable_fraction` per class.
In a partly saturated class the remaining (unsaturated) pixels are the lower-DPM ones, so predictions there lean thin.

### Hold-out scores

Two versions are reported. The **headline** is the error of a **single bootstrap fit** on its out-of-bag pixels
(median over rounds, with the 95% range); this is the error to expect from the one fit you use.
The **bagged** score averages each pixel's out-of-bag predictions over many rounds and is slightly optimistic.

## 3. Settings: where to change things

All defaults are in the **DEFAULT SETTINGS** block at the top of `fit_thickness_from_dpm.m`.
Edit them there, or override any of them per call, e.g. `fit_thickness_from_dpm(data, edges_mm, 'MaxPixelsPerClass', 50)`.

| Setting (name-value) | Default | Meaning |
|---|---|---|
| `MinPixelsPerClass` | 10 | **Stops with an error** if any class has fewer pixels. The message lists the classes, suggests merged edges, and gives the line to edit |
| `WarnPixelsPerClass` | 30 | Warns (and continues) for classes with fewer pixels |
| `MaxPixelsPerClass` | 5000 | Random cap per class so no class dominates |
| `NumBootstrap` | 1000 | Out-of-bag bootstrap rounds |
| `FitTarget` | `'class_medians'` | `'class_medians'` or `'pixels'` |
| `Coverage` | 0.95 | Coverage of the single-pixel prediction interval |
| `SaturationDpm` | 0.99 | DPM at/above this counts as saturated ("the ruler has run out": a lot of change, amount unknown) |
| `SaturationWarnFraction` | 0.10 | Warns if more than this fraction of any class is saturated |
| `SaturationStopFraction` | 0.50 | **Stops** if more than this fraction of any class is saturated (its median is then saturated) |
| `DpmClipMargin` | 0.005 | DPM at/beyond floor or ceiling is clipped this far inside them during fitting |
| `RandomSeed` | 42 | All random steps are reproducible |
| `OutputDirectory`, `OutputBasename`, `MakePlot` | `'.'`, `'dpm_thickness'`, true | Figures (PNG at 150 dpi and PDF) |

If the data are too small, merge thin classes by removing an edge from `isopach_edges_mm` in your call
(the error message suggests which), or lower `MinPixelsPerClass`.

## 4. Predicting thickness from DPM

```matlab
predict_thickness_from_dpm(results, [0.2 0.4 0.6])                 % prints a table
prediction_table = predict_thickness_from_dpm(results, dpm_values);
[~, maps] = predict_thickness_from_dpm(results, dpm_image);         % map in -> maps out, same shape
```

| Output | Meaning |
|---|---|
| `thickness_mm` | Point estimate: a typical (median-like) thickness, **not** the mean |
| `ci_low_mm`, `ci_high_mm` | 95% **confidence** interval: where the calibration curve lies. Narrow |
| `pi_low_mm`, `pi_high_mm` | **Prediction** interval for **one pixel**: where its thickness could lie. Wide (≈ ×0.24 to ×3.7 in the example) |
| `flag` | 0 ok · 1 extrapolation (DPM outside the fitted DPM range) · 2 below floor, thickness not resolvable (NaN) · 3 above ceiling, saturated (NaN) · 4 input NaN · **5 thickness outside the calibrated isopach range** (value kept, e.g. 2000 mm when the top isopach is 500 mm, but not supported by data: read it as "at least ~500 mm") |

Use the PI for single-pixel statements, the CI for the average relationship.
With two outputs (`[~, maps] = …`) the table is **not** built, so a full DPM scene does not run out of memory.

## 5. Preprocessing: from rasters to the 2-column table

1. **DPM raster.** Single-band GeoTIFF, values 0–1 (if 0–100 or 0–255, use `--dpm_divide_by`).
   Set invalid pixels to NaN/nodata: water, layover/shadow, low-coherence masks, anything outside the study area.
   The coherence pair must bracket the ash fall. Rain, vegetation change, or other events inside the
   same window also lower coherence and will leak into the result.
2. **Thickness raster in mm.** Convert the isopach map to a GeoTIFF whose pixel value is ash thickness in **mm**
   (cm × 10). Any grid and CRS is fine as long as the CRS is defined; the script resamples it onto the DPM grid.
   If your isopachs are **class polygons** (area between two contours), add a column `thickness_mm` holding
   the geometric mean of the class edges, `sqrt(lower * upper)`, then rasterize:
   ```bash
   gdal_rasterize -a thickness_mm -a_nodata -9999 -ot Float32 -te XMIN YMIN XMAX YMAX -tr XRES YRES isopach_classes.shp isopach_thickness_mm.tif
   ```
   (take the extent and resolution from `gdalinfo dpm.tif`). If you only have contour **lines**,
   build class polygons or interpolate a thickness surface first.
3. **Choose the class edges** (mm, increasing, all > 0; use 0.1, not 0). Use your isopach intervals.
   Include a very-thin or ash-free reference class if you can: it pins down the DPM noise floor.
4. **Build the table** (reads the rasters in blocks, so large scenes are fine):
   ```bash
   python preprocess_dpm_isopach.py --dpm dpm.tif --isopach isopach_thickness_mm.tif --edges 0.1 1 5 10 50 100 200 500 --samples_per_class 2000 --output dpm_thickness.csv
   ```
   It draws up to the same number of random pixels from each class. Use the **same edges** here and in MATLAB.
5. **Check it.** `head dpm_thickness.csv` should show two columns (`dpm`, `thickness_mm`); look at the per-class
   pixel counts printed by the script.

## 6. Caveats (read before interpreting)

- **Spatial correlation.** Neighbouring DPM pixels are correlated. The bootstrap resamples pixels as if independent,
  so the confidence band, prediction interval and hold-out scores are **optimistic** for a new area.
  A 2-column input has no coordinates, so a spatial-block hold-out is not possible.
- **DPM measures coherence loss, not ash.** Ash can decorrelate a surface without damage, and other changes
  (rain, vegetation, construction) raise DPM without ash. Calibrate per scene/pair; do not transfer blindly.
- **Floor and ceiling.** Below the floor DPM says nothing about thickness (flag 2); near the ceiling the curve is
  steep and thickness is poorly constrained: in the synthetic example DPM 0.84-0.905 already gives 0.6-12 m,
  far beyond the 500 mm top isopach (flag 5). If the ceiling hits its limit (1.0) the function warns: the data show no
  saturation and anything above the highest observed DPM is extrapolation.
- **Log10 thickness.** Errors are multiplicative ("×2"), suited to thickness spanning orders of magnitude.
  The prediction is a typical (median-like) thickness; summing it over an area underestimates the mean volume.
- **Sampling design.** Pixels are sampled equally per thickness class and thickness is then predicted from DPM, so
  the hold-out scores and the prediction interval are valid for pixels drawn the **same way** (equal per class).
  Applied to every pixel of a map, where thin ash usually covers far more area, single-pixel predictions are pulled
  toward the classes that were over-sampled. The pixels fit is more affected than the medians fit.
- **Thickness is the isopach value, not a measurement.** For class polygons every pixel gets the class value, so the
  hold-out error and prediction interval are relative to the isopach, not to the true thickness. The within-class
  spread (up to a decade for 0.1–1 mm) is not included.
- **Prediction interval width is constant in log10 thickness.** `results.prediction_interval.class_coverage`
  shows classes where it covers less than intended (the function warns when a class is more than 5 points below
  the target coverage, i.e. below 90% for 95%). It is calibrated only on pixels the prediction would actually predict
  (DPM strictly between floor and ceiling).
- **Isopach thickness is itself interpolated** and uncertain, especially in the outer classes.

## 7. Synthetic example results (`make_example_dpm_csv.m`, 50 pixels per class)

Truth: t50 = 30 mm, k = 2.0, floor = 0.05, ceiling = 0.90.

| | Value |
|---|---|
| Fitted (medians) | t50 = 34.0 mm, k = 1.94, floor = 0.047, ceiling = 0.911 |
| Fitted vs noise-free generator curve, DPM 0.1–0.85 | within 1–11%, inside the 95% CI at all 7 test DPMs (the noise-free curve is an approximate target; equal-per-class sampling shifts the exact one slightly) |
| Out-of-bag RMSE, single fit | 0.297 log10 mm (×1.98), 95% of fits 0.259–0.430; 69% within ×2 |
| Independent test pixels (new seed) | RMSE 0.288 log10 mm, 70% within ×2, PI coverage 96.2% (94.8–97.4% per class) |
