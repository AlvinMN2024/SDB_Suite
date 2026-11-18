# Changelog

## 1.1.0

- Aggregate sounding records to one median target and predictor vector per raster cell before validation and final fitting.
- Replace OOB metrics with grouped spatial cross-validation; allow user-selected block size and optional distance buffer, and label zero-buffer limitations.
- Save cross-validation predictions, folds, block IDs, point counts, raster grid details, selected features, and model settings in the joblib payload.
- Add depth-stratified metrics, visible/all-feature comparison, same-fold Stumpf-style and Lyzenga-style baselines, permutation importance, and optional RF sensitivity diagnostics.
- Add explicit training/prediction water-level inputs relative to MLLW and convert generated depths to negative MLLW elevations.
- Map saved feature names to stack bands explicitly, reject missing/unsupported model inputs, respect raster nodata, and replace the fixed NIR land cutoff with an optional aligned water mask.
- Remove automated precision and IHO suitability labels and revise the plugin description and user guide to describe validation limits accurately.
- Correct recognized predictor aliases for the `rhow_833`/`rhow_883` label and Nechad TUR/SPM column suffixes.
