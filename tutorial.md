# SDB Production Suite: field-to-raster checklist

This guide covers the plugin workflow. It does not prescribe ACOLITE settings: those depend on the scene, water, atmospheric conditions, and processing objective. Keep the processing settings and source scene with each analysis.

Use the output as a local reconnaissance surface and supplement to soundings and expert review. This release does not establish it as the sole basis for under-keel-clearance decisions or official chart depths; independently check critical shallow areas.

## 1. Prepare the predictor stack and CSV

Use the processed raster stack from which the CSV predictors were sampled. The trainer uses the raster to assign soundings to exact cells and blocks; the predictor values themselves are read from the CSV. Keep the stack and CSV from the same processing run.

The CSV requires:

- `X` and `Y` in one known CRS;
- one depth field named `MLLW`, `mllw`, `depth`, `Depth`, `Z`, or `z`;
- numeric predictor columns such as `rhow_443`, `rhow_492`, `rhow_560`, `rhow_665`, `rhow_704`, `rhow_833` or `rhow_883`, `rhow_865`, `TUR_Nechad` or `TUR_Nechad2016_665`, and `SPM_Nechad` or `SPM_Nechad2016_665`.

Use one depth sign convention throughout. Positive depths below MLLW and negative elevations below MLLW are both accepted; mixed signs are rejected. The shallow and deep limits in the tool are entered as positive depths in metres.

Before training, verify the CSV CRS, raster CRS, raster dimensions, pixel size, and that the values in the CSV were sampled from this stack. The plugin checks coordinates against the raster extent and maps them to cells, but does not compare each CSV predictor value with a fresh raster sample.

Confirm that every stack band is on the same grid. A band originally measured at 20 m or 60 m and resampled by ACOLITE to 10 m still has coarser native spatial support; record that resampling and avoid interpreting it as new 10 m detail.

## 2. Train and validate

Use the **1. Train Bathymetry Model (RF)** algorithm:

1. Select the CSV, predictor grid, and CSV coordinate CRS.
2. Set the spatial block size in the raster CRS units. A variogram or another spatial-dependence analysis can inform this choice; the plugin does not calculate one.
3. Set the number of folds. The number of occupied blocks must be at least this large.
4. Set a buffer distance if justified. It removes training pixel centers within that distance of each held-out fold. A zero buffer can leave training points close to held-out blocks, so the log warns that those results are not fully independent.
5. Choose visible-only or all available features. The trainer reports both feature-set CV summaries and uses the selected feature set for the final saved model.
6. Set the image-time water level relative to MLLW if known. Positive means water above MLLW. If unknown, zero is an explicit no-correction assumption, not evidence that tide was zero.
7. Review pooled metrics, depth-stratified metrics, baseline diagnostics, permutation importance, and any warnings before keeping the model.

Soundings assigned to the same raster cell are reduced to median predictor and depth values before folds are made. Report both the original sounding count and the occupied-pixel count. Fold metrics are pooled from held-out predictions and are not training-set fit metrics.

The Stumpf-style and Lyzenga-style outputs are diagnostic linear baselines calibrated on the same folds. Treat them as comparisons for this plugin run, not as reference-grade implementations. Permutation importance measures predictive change for these held-out folds; it does not show causal or physical importance.

## 3. Generate a raster

Use the **2. Generate Bathymetry Map (RF)** algorithm:

1. Select the predictor raster stack and the saved model payload.
2. Match the band indices to the stack. Defaults follow the historical suite order: visible bands, TUR, SPM, red edge, NIR, NIR. Verify every index; file order is not guaranteed by ACOLITE export.
3. Enter prediction-time water level relative to MLLW, positive above. The model output is adjusted by subtracting this value from positive depth, then written as negative MLLW elevation.
4. Optionally provide a water mask aligned exactly to the predictor raster. Values above zero are treated as water. Without a mask, the plugin masks nodata/non-finite predictors only; it does not identify land with a fixed NIR cutoff.
5. Leave depth stretch at 1 and offset at 0 unless you have a separately justified adjustment. These controls are applied after prediction and are not included in the saved CV metrics.

The maximum output-depth setting masks predictions deeper than its positive threshold. Set it to zero to disable that mask. It does not improve model validation or make extrapolated predictions reliable.

## 4. Inspect and report

Compare the raster with the training soundings and, where possible, soundings not used to train the model, another date, or another survey. Check coast/land masking, nodata edges, shallow/deep strata, spatial patterns in residuals, and regions outside the training range.

When reporting a result, state at least: scene/date and processing, raster grid and predictor set, reference dataset and datum, depth interval, pixel and sounding counts, block size, buffer distance, folds, metrics, and known reference uncertainty. Distinguish internal spatial CV from independent-site or cross-date evaluation. Do not label output hydrographic-grade, certified, or IHO S-44 compliant based on this plugin's metrics.

## 5. Known scope

This release does not estimate residual variogram range or optical extinction depth, model survey uncertainty, retrieve tide automatically, run a cross-date test, or certify navigation fitness. A tide correction does not account for echo-sounder heave, pitch/roll, sound speed, draft, horizontal positioning, or survey-to-image time gap. Keep those uncertainties in the interpretation.
