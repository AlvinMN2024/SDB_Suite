# SDB Production Suite

A QGIS Processing provider for fitting Random Forest models to satellite reflectance and sounding data, then applying those models to raster stacks. This release focuses on transparent local analysis and review.

The intended role is local reconnaissance and supplementary bathymetry review: visualizing likely depth patterns and helping direct follow-up checks. Use it within conditions represented by the training data and review it against soundings, vessel observations, and local knowledge. This release does not support using its output as the sole basis for under-keel-clearance decisions or official chart depths; independently verify critical shallow areas.

The plugin produces estimates from the supplied predictors and soundings. Its metrics describe agreement with those reference data under the selected validation design. They do not establish navigation safety, survey certification, or compliance with IHO S-44.

## Requirements

- QGIS 3.22 through 4.x (metadata currently permits versions through 4.99).
- The QGIS Python environment must provide `numpy`, `pandas`, `scikit-learn`, and `joblib`.
- The plugin uses QGIS/GDAL raster bindings; it does not require `rasterio`.

The way to install Python packages depends on the QGIS distribution. Install them into the Python environment used by QGIS, not an unrelated system Python.

## Workflow

1. Prepare a CSV of soundings with numeric `X`, `Y` columns, a depth column named `MLLW`, `mllw`, `depth`, `Depth`, `Z`, or `z`, and predictor columns.
2. In **1. Train Bathymetry Model (RF)**, select the CSV, the raster grid used to sample its predictors, and the CSV coordinate CRS. Choose block and buffer distances in the raster CRS units.
3. Review the Processing log and saved model payload. The payload includes metrics, out-of-fold predictions, exact fold membership, complete fitted RF settings, baseline settings, row/pixel counts, and grid details.
4. In **2. Generate Bathymetry Map (RF)**, select the predictor stack and model. Check every band index against the actual stack. Set the image-time water level relative to MLLW if known, and optionally provide an aligned water mask.
5. Inspect the resulting negative-elevation raster against soundings, coastlines, independent checks, and local knowledge before using it.

## Trainer details

### Soundings and pixels

Coordinates are transformed into the selected raster CRS, assigned to cells using the raster geotransform, and aggregated to one record per occupied cell. Predictor values and positive depths are reduced by the median. This prevents repeated soundings in one Sentinel-2 cell from appearing as separate model records. The cell's sounding count is retained in the model payload.

All predictor bands must already be aligned on the input stack grid. Resampling a band with a coarser native resolution onto a 10 m grid does not create 10 m information; retain the ACOLITE resampling choices and original band resolutions in the analysis record.

Depth values must use one sign convention throughout the CSV. Negative elevations are converted to positive depth internally. The configured shallow and deep limits are positive depths in metres.

### Spatial validation

The trainer uses `GroupKFold` over map-coordinate blocks. A buffer can exclude training pixels within the selected distance of held-out pixel centers. The default buffer is zero, which means blocks are held out but nearby training cells may remain close to a test block. In that case, the tool explicitly warns against describing the result as fully independent.

The plugin does not estimate a residual variogram or choose a defensible block size automatically. Select block and buffer sizes based on spatial dependence and intended prediction scale, and report sensitivity to those choices. Blocked cross-validation is still internal validation on one dataset; it does not replace an independent survey or cross-date test.

Reported pooled metrics are R² (when defined), RMSE, MAE, and bias (`prediction - reference`). They are calculated from one out-of-fold prediction per occupied raster cell, converted back to MLLW. The trainer also reports depth strata, visible-only versus all-available feature sets, two same-fold baseline diagnostics, and held-out permutation importance for the selected feature set. Baselines are labelled “style” because they are simple calibrated implementations, not certified reference implementations.

### Model settings

The Random Forest uses 200 trees, `random_state=42`, `n_jobs=-1`, and `oob_score=False`. The user can set `max_depth` (0 means unlimited), `min_samples_leaf`, and `max_features` (a fraction from 0.01 to 1.0). Other estimator values use the installed scikit-learn defaults. The payload stores the full fitted parameter dictionary and scikit-learn version. Optional sensitivity diagnostics compare a fixed descriptive grid; they do not automatically select parameters.

Joblib model files are Python serialization artifacts, not long-term interchange formats. Keep the training environment details and scikit-learn version with the model; loading across substantially different Python or scikit-learn versions may fail or behave differently.

The target is `log(image-time positive depth + 0.1)`. Training tide is entered as water level relative to MLLW, positive above MLLW. The model payload stores that value. Tide is not inferred from the image or obtained from a tide service.

## Generator details

Model feature names are stored in the payload and matched explicitly to the selected stack bands. Supported names include the visible/red-edge/NIR `rhow_*` columns, `ln_665`, and `TUR_Nechad` / `SPM_Nechad` with or without the `2016_665` suffix. Older payloads without feature names use a warned legacy ordering and should be checked against their training setup.

The generator converts the model's positive image-time depth estimate to MLLW by subtracting the supplied prediction-time water level, then writes negative elevations. Zero tide means no tide adjustment. A provided water mask must have exactly matching dimensions, geotransform, and CRS; positive values denote water. With no water mask, only nodata and non-finite predictors are masked. There is no universal reflectance-threshold land mask.

Depth stretch and offset remain available as optional post-processing controls. Their defaults (1 and 0) leave predictions unchanged; using them changes the output beyond the saved validation metrics.

## Limits to report

- No accuracy value should be presented as validated solely because it appears in this plugin's output. State the validation design, block size, buffer, sample/pixel count, depth range, and reference-data limitations.
- The plugin does not yet provide a cross-date or external-test workflow.
- It reports depth-stratified errors but does not estimate optical extinction depth from the supplied data.
- Training and prediction tide corrections require a defensible water level and datum relationship. A tide value alone does not correct survey soundings for heave, sound speed, draft, positioning, or other reference uncertainty.
- The model estimates bathymetry only over conditions represented by its training data and predictors. Examine extrapolation, masks, and artifacts manually.

## Files

- `sdb_trainer_alg.py` — pixel aggregation, spatial validation, diagnostics, and model payload creation.
- `sdb_generator_alg.py` — feature-aware, windowed raster prediction.
- `sdb_provider.py` — QGIS Processing registration.
- `tutorial.md` — operational checklist and interpretation guidance.
- `SDB_Manual.pdf` — printable copy of the release guidance.
- `CHANGELOG.md` — changes in this release.

## License and project

GPL-2.0-or-later. Source and issue tracker: [SDB Suite on GitHub](https://github.com/AlvinMN2024/SDB_Suite).
