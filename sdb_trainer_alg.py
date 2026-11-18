
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsProcessingException
)

# --- Safe Dependency Imports ---
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import NearestNeighbors
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


class SDBTrainerAlgorithm(QgsProcessingAlgorithm):
    CSV_INPUT = 'CSV_INPUT'
    GRID_RASTER = 'GRID_RASTER'
    CSV_CRS = 'CSV_CRS'
    BLOCK_SIZE = 'BLOCK_SIZE'
    BUFFER_DISTANCE = 'BUFFER_DISTANCE'
    N_FOLDS = 'N_FOLDS'
    FEATURE_SET = 'FEATURE_SET'
    DEPTH_BIN_WIDTH = 'DEPTH_BIN_WIDTH'
    STUMPF_N = 'STUMPF_N'
    RUN_RF_SENSITIVITY = 'RUN_RF_SENSITIVITY'
    TRAINING_TIDE = 'TRAINING_TIDE'
    RF_MAX_DEPTH = 'RF_MAX_DEPTH'
    RF_MIN_SAMPLES_LEAF = 'RF_MIN_SAMPLES_LEAF'
    RF_MAX_FEATURES = 'RF_MAX_FEATURES'
    SHALLOW_MIN = 'SHALLOW_MIN'
    SHALLOW_MAX = 'SHALLOW_MAX'
    MODEL_OUTPUT = 'MODEL_OUTPUT'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return SDBTrainerAlgorithm()

    def name(self):
        return 'sdb_trainer'

    def displayName(self):
        return self.tr('1. Train Bathymetry Model (RF)')

    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        return QIcon(icon_path) if os.path.exists(icon_path) else super().icon()

    def shortHelpString(self):
        return self.tr(
            "<h2>SDB Production Suite: Model Trainer</h2>"
            "<p>Trains a Random Forest (RF) model using satellite reflectance data and depth soundings.</p>"
            "<h3>Usage Notes:</h3>"
            "<ul>"
            "  <li>Provide a CSV with numeric X/Y coordinates, surface reflectances (rhow_*), and target depths.</li>"
            "  <li>Select the raster grid used to sample the CSV and the CRS of its X/Y coordinates.</li>"
            "  <li>Soundings are aggregated by raster pixel before spatial-block cross-validation. Choose block size using spatial dependence in your data; a buffer can remove nearby training pixels around each held-out fold.</li>"
            "  <li>The plugin does not estimate a residual variogram. Metrics without a positive buffer are spatially grouped, but nearby train/test blocks may still be correlated.</li>"
            "  <li>Training tide is stored in the model; prediction tide is set in the generator. Both are water levels relative to MLLW, positive above. Zero means no correction.</li>"
            "  <li>Depth input may be consistently positive or negative; configured shallow/deep limits are positive depths in metres.</li>"
            "  <li>RF: 200 trees, random_state=42, OOB disabled; set max_depth (0=unlimited), min_samples_leaf, and max_features.</li>"
            "  <li>Metrics are pooled from out-of-fold predictions on occupied pixels. They do not establish independent-site performance or hydrographic certification.</li>"
            "</ul>"
        )

    def group(self):
        return self.tr('SDB Suite')

    def groupId(self):
        return 'sdb_suite'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.CSV_INPUT, self.tr('Input Control Soundings CSV'), extension='csv'))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.GRID_RASTER,
            self.tr('Raster grid used to sample the CSV reflectances')
        ))
        self.addParameter(QgsProcessingParameterCrs(
            self.CSV_CRS,
            self.tr('CRS of the CSV X/Y coordinate columns')
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.BLOCK_SIZE,
            self.tr('Spatial block size (units of raster CRS; starting value only)'),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=100.0,
            minValue=0.001
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.BUFFER_DISTANCE,
            self.tr('Spatial buffer around held-out pixels (raster CRS units; 0 disables)'),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.0,
            minValue=0.0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_FOLDS,
            self.tr('Number of spatial CV folds'),
            type=QgsProcessingParameterNumber.Type.Integer,
            defaultValue=5,
            minValue=2
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.FEATURE_SET,
            self.tr('RF feature set for final model'),
            options=[
                self.tr('Visible bands only'),
                self.tr('All available predictors')
            ],
            defaultValue=1
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.DEPTH_BIN_WIDTH,
            self.tr('Depth-stratum width (m)'),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=5.0,
            minValue=0.1
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.STUMPF_N,
            self.tr('Stumpf ratio scale factor (diagnostic baseline)'),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=1000.0,
            minValue=0.001
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.RUN_RF_SENSITIVITY,
            self.tr('Run RF hyperparameter sensitivity diagnostics'),
            defaultValue=False
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.TRAINING_TIDE,
            self.tr('Image-time water level relative to MLLW (m; positive above MLLW)'),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.RF_MAX_DEPTH,
            self.tr('RF max_depth (0 means unlimited)'),
            type=QgsProcessingParameterNumber.Type.Integer,
            defaultValue=0,
            minValue=0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.RF_MIN_SAMPLES_LEAF,
            self.tr('RF min_samples_leaf'),
            type=QgsProcessingParameterNumber.Type.Integer,
            defaultValue=1,
            minValue=1
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.RF_MAX_FEATURES,
            self.tr('RF max_features fraction (0 < value <= 1)'),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=1.0,
            minValue=0.01,
            maxValue=1.0
        ))
        #self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MIN, self.tr('Min Depth (m, positive)'), defaultValue=0.5, type=QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MIN, self.tr('Min Depth (m, positive)'), defaultValue=0.5, type=QgsProcessingParameterNumber.Type.Double))
        #self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MAX, self.tr('Max Depth (m, positive)'), defaultValue=20.0, type=QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MAX, self.tr('Max Depth (m, positive)'), defaultValue=20.0, type=QgsProcessingParameterNumber.Type.Double))
        self.addParameter(QgsProcessingParameterFileDestination(self.MODEL_OUTPUT, self.tr('Save Model (.pkl)'), 'Pickle (*.pkl)'))

    def _pixel_indices(self, df, grid_layer, csv_crs, context):
        """Map CSV coordinates to cells of the exact predictor raster grid."""
        if not HAS_GDAL:
            raise QgsProcessingException(self.tr("GDAL Python bindings are required for grid-aware validation."))
        if grid_layer is None or not grid_layer.isValid():
            raise QgsProcessingException(self.tr("Select a valid raster grid."))
        if not csv_crs.isValid() or not grid_layer.crs().isValid():
            raise QgsProcessingException(self.tr("Select valid CRSs for the CSV and raster grid."))
        if grid_layer.crs().isGeographic():
            raise QgsProcessingException(
                self.tr("The raster CRS is geographic. Reproject the stack to a projected CRS before using map-unit spatial blocks.")
            )
        if 'X' not in df.columns or 'Y' not in df.columns:
            raise QgsProcessingException(self.tr("CSV must contain numeric X and Y coordinate columns."))

        try:
            x_values = pd.to_numeric(df['X'], errors='raise').to_numpy(dtype=float)
            y_values = pd.to_numeric(df['Y'], errors='raise').to_numpy(dtype=float)
        except Exception as e:
            raise QgsProcessingException(f"CSV X/Y coordinates must be numeric: {str(e)}")
        if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
            raise QgsProcessingException(self.tr("CSV X/Y coordinates contain non-finite values."))

        ds = gdal.Open(grid_layer.dataProvider().dataSourceUri())
        if ds is None:
            raise QgsProcessingException(self.tr("Could not open the selected raster grid via GDAL."))
        geotransform = ds.GetGeoTransform()
        width, height = ds.RasterXSize, ds.RasterYSize
        ds = None

        determinant = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]
        if abs(determinant) < 1e-12:
            raise QgsProcessingException(self.tr("Raster grid transform is not invertible."))

        coord_transform = QgsCoordinateTransform(csv_crs, grid_layer.crs(), context.transformContext())
        pixel_rows = np.empty(len(df), dtype=np.int64)
        pixel_cols = np.empty(len(df), dtype=np.int64)

        try:
            for i, (x, y) in enumerate(zip(x_values, y_values)):
                point = coord_transform.transform(QgsPointXY(x, y))
                dx = point.x() - geotransform[0]
                dy = point.y() - geotransform[3]
                col_f = (dx * geotransform[5] - dy * geotransform[2]) / determinant
                row_f = (dy * geotransform[1] - dx * geotransform[4]) / determinant
                pixel_cols[i] = int(np.floor(col_f))
                pixel_rows[i] = int(np.floor(row_f))
        except Exception as e:
            raise QgsProcessingException(f"Could not transform CSV coordinates to the raster CRS: {str(e)}")

        outside = (
            (pixel_cols < 0) | (pixel_cols >= width) |
            (pixel_rows < 0) | (pixel_rows >= height)
        )
        if np.any(outside):
            raise QgsProcessingException(
                self.tr(f"{int(np.count_nonzero(outside))} retained CSV points fall outside the selected raster grid. Check the coordinate CRS and raster choice.")
            )
        return pixel_rows, pixel_cols, geotransform, width, height

    @staticmethod
    def _regression_metrics(observed, predicted):
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        valid = np.isfinite(observed) & np.isfinite(predicted)
        observed, predicted = observed[valid], predicted[valid]
        if len(observed) == 0:
            return {'count': 0, 'r2': None, 'rmse': None, 'mae': None, 'bias': None}
        r2 = None
        if len(observed) >= 2 and np.var(observed) > 0:
            r2 = float(r2_score(observed, predicted))
        return {
            'count': int(len(observed)),
            'r2': r2,
            'rmse': float(np.sqrt(mean_squared_error(observed, predicted))),
            'mae': float(mean_absolute_error(observed, predicted)),
            'bias': float(np.mean(predicted - observed))
        }

    @staticmethod
    def _depth_metrics(observed, predicted, min_depth, max_depth, bin_width):
        edges = list(np.arange(min_depth, max_depth, bin_width, dtype=float))
        if not edges or edges[0] != min_depth:
            edges.insert(0, float(min_depth))
        edges.append(float(max_depth))
        edges = sorted(set(edges))
        if len(edges) == 1:
            edges.append(edges[0] + bin_width)

        results = {}
        for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            if i == len(edges) - 2:
                mask = (observed >= lower) & (observed <= upper)
            else:
                mask = (observed >= lower) & (observed < upper)
            results[f'{lower:g} to {upper:g} m'] = SDBTrainerAlgorithm._regression_metrics(
                observed[mask], predicted[mask]
            )
        return results

    def _run_rf_cv(self, X, y_log, y_observed_mllw, training_tide, groups,
                   splits, model_parameters, feature_names, calculate_permutation=False):
        oof_prediction = np.full(len(y_log), np.nan, dtype=float)
        fold_ids = np.full(len(y_log), -1, dtype=int)
        importance = {name: [] for name in feature_names}

        for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
            model = RandomForestRegressor(
                n_estimators=model_parameters.get('n_estimators', 200),
                max_depth=model_parameters.get('max_depth'),
                min_samples_leaf=model_parameters.get('min_samples_leaf', 1),
                max_features=model_parameters.get('max_features', 1.0),
                oob_score=False,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X[train_idx], y_log[train_idx])
            pred_image_depth = np.exp(model.predict(X[test_idx])) - 0.1
            pred_mllw = pred_image_depth - training_tide
            oof_prediction[test_idx] = pred_mllw
            fold_ids[test_idx] = fold_number

            if calculate_permutation:
                baseline_rmse = np.sqrt(mean_squared_error(y_observed_mllw[test_idx], pred_mllw))
                rng = np.random.RandomState(4200 + fold_number)
                for column, name in enumerate(feature_names):
                    deltas = []
                    for _ in range(5):
                        permuted = X[test_idx].copy()
                        permuted[:, column] = rng.permutation(permuted[:, column])
                        perm_pred = np.exp(model.predict(permuted)) - 0.1 - training_tide
                        perm_rmse = np.sqrt(mean_squared_error(y_observed_mllw[test_idx], perm_pred))
                        deltas.append(float(perm_rmse - baseline_rmse))
                    importance[name].append(float(np.mean(deltas)))

        if not np.all(np.isfinite(oof_prediction)):
            raise QgsProcessingException(self.tr("Spatial CV did not produce a prediction for every pixel."))
        importance_summary = {
            name: float(np.mean(values)) for name, values in importance.items() if values
        }
        return oof_prediction, fold_ids, importance_summary

    def _run_linear_baseline(self, method, X_reflectance, y_image_depth,
                             y_observed_mllw, training_tide, splits, stumpf_n=1000.0):
        oof_prediction = np.full(len(y_image_depth), np.nan, dtype=float)
        for train_idx, test_idx in splits:
            if method == 'stumpf_ratio':
                blue_train = np.maximum(X_reflectance[train_idx, 0], 1e-6)
                green_train = np.maximum(X_reflectance[train_idx, 1], 1e-6)
                blue_test = np.maximum(X_reflectance[test_idx, 0], 1e-6)
                green_test = np.maximum(X_reflectance[test_idx, 1], 1e-6)
                train_den = np.log(np.maximum(stumpf_n * green_train, 1e-6))
                test_den = np.log(np.maximum(stumpf_n * green_test, 1e-6))
                train_den = np.where(np.abs(train_den) < 1e-8, 1e-8, train_den)
                test_den = np.where(np.abs(test_den) < 1e-8, 1e-8, test_den)
                x_train = (np.log(np.maximum(stumpf_n * blue_train, 1e-6)) / train_den).reshape(-1, 1)
                x_test = (np.log(np.maximum(stumpf_n * blue_test, 1e-6)) / test_den).reshape(-1, 1)
            elif method == 'lyzenga_log_linear':
                band_minima = np.min(X_reflectance[train_idx], axis=0)
                x_train = np.log(np.maximum(X_reflectance[train_idx] - band_minima + 1e-6, 1e-6))
                x_test = np.log(np.maximum(X_reflectance[test_idx] - band_minima + 1e-6, 1e-6))
            else:
                raise ValueError(f'Unknown baseline method: {method}')

            model = LinearRegression()
            model.fit(x_train, y_image_depth[train_idx])
            oof_prediction[test_idx] = model.predict(x_test) - training_tide

        if not np.all(np.isfinite(oof_prediction)):
            raise QgsProcessingException(self.tr(f"{method} did not produce a prediction for every pixel."))
        return oof_prediction

    @staticmethod
    def _available_features(df):
        """Return visible and full predictor lists while preserving CSV column names."""
        visible = ['ln_665', 'rhow_443', 'rhow_492', 'rhow_560', 'rhow_665']
        all_features = list(visible)
        for alias_group in (
            ('rhow_704',),
            ('rhow_833', 'rhow_883'),
            ('rhow_865',),
            ('TUR_Nechad', 'TUR_Nechad2016_665'),
            ('SPM_Nechad', 'SPM_Nechad2016_665')
        ):
            matches = [name for name in alias_group if name in df.columns]
            if len(matches) > 1:
                raise QgsProcessingException(
                    f"CSV contains multiple aliases for one predictor: {matches}. Keep one column or resolve them explicitly."
                )
            if matches:
                all_features.append(matches[0])
        visible = [name for name in visible if name in df.columns]
        all_features = [name for name in all_features if name in df.columns]
        return visible, all_features

    def processAlgorithm(self, parameters, context, feedback):
        if not HAS_SKLEARN:
            raise QgsProcessingException(self.tr("scikit-learn and joblib are required in the QGIS Python environment."))

        csv_path = self.parameterAsFile(parameters, self.CSV_INPUT, context)
        grid_layer = self.parameterAsRasterLayer(parameters, self.GRID_RASTER, context)
        csv_crs = self.parameterAsCrs(parameters, self.CSV_CRS, context)
        block_size = self.parameterAsDouble(parameters, self.BLOCK_SIZE, context)
        buffer_distance = self.parameterAsDouble(parameters, self.BUFFER_DISTANCE, context)
        n_folds = self.parameterAsInt(parameters, self.N_FOLDS, context)
        feature_choice = self.parameterAsEnum(parameters, self.FEATURE_SET, context)
        depth_bin_width = self.parameterAsDouble(parameters, self.DEPTH_BIN_WIDTH, context)
        stumpf_n = self.parameterAsDouble(parameters, self.STUMPF_N, context)
        run_sensitivity = self.parameterAsBool(parameters, self.RUN_RF_SENSITIVITY, context)
        training_tide = self.parameterAsDouble(parameters, self.TRAINING_TIDE, context)
        max_depth_param = self.parameterAsInt(parameters, self.RF_MAX_DEPTH, context)
        min_samples_leaf = self.parameterAsInt(parameters, self.RF_MIN_SAMPLES_LEAF, context)
        max_features = self.parameterAsDouble(parameters, self.RF_MAX_FEATURES, context)
        model_path = self.parameterAsFileOutput(parameters, self.MODEL_OUTPUT, context)
        s_min = self.parameterAsDouble(parameters, self.SHALLOW_MIN, context)
        s_max = self.parameterAsDouble(parameters, self.SHALLOW_MAX, context)

        if s_max <= s_min:
            raise QgsProcessingException(self.tr("Maximum depth must be greater than minimum depth."))
        if block_size <= 0 or buffer_distance < 0 or n_folds < 2 or depth_bin_width <= 0 or stumpf_n <= 0:
            raise QgsProcessingException(self.tr("Block size, depth-bin width, and Stumpf scale must be positive; buffer cannot be negative; folds must be at least 2."))
        if min_samples_leaf < 1 or not 0 < max_features <= 1:
            raise QgsProcessingException(self.tr("RF min_samples_leaf must be >= 1 and max_features must be in (0, 1]."))

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise QgsProcessingException(f"Failed to read CSV file: {str(e)}")
        if df.empty:
            raise QgsProcessingException(self.tr("Input CSV has no rows."))

        depth_col = next((c for c in ['MLLW', 'mllw', 'depth', 'Depth', 'Z', 'z'] if c in df.columns), None)
        if depth_col is None:
            raise QgsProcessingException(self.tr("CSV must contain one of: MLLW, mllw, depth, Depth, Z, or z."))
        if 'rhow_665' in df.columns and 'ln_665' not in df.columns:
            df['ln_665'] = np.log(np.maximum(pd.to_numeric(df['rhow_665'], errors='coerce'), 1e-6))

        visible_features, all_features = self._available_features(df)
        if not visible_features:
            raise QgsProcessingException(self.tr("No visible reflectance features were found. Expected rhow_443/492/560/665 (or ln_665)."))
        if not all_features:
            raise QgsProcessingException(self.tr("No usable predictors were found in the CSV."))
        if feature_choice == 0:
            selected_features = visible_features
            selected_set_name = 'visible_only'
        else:
            selected_features = all_features
            selected_set_name = 'all_available'

        if not {'X', 'Y'}.issubset(df.columns):
            raise QgsProcessingException(self.tr("CSV must contain numeric X and Y coordinate columns."))
        required_fields = list(dict.fromkeys(['X', 'Y', depth_col] + all_features))
        input_row_count = len(df)
        for field in required_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        depth_values_all = df[depth_col].dropna().to_numpy(dtype=float)
        if np.any(depth_values_all < 0) and np.any(depth_values_all > 0):
            raise QgsProcessingException(self.tr("Depth column mixes positive depths and negative elevations. Convert it to one convention before training."))
        valid_before_range = len(df.dropna(subset=required_fields))
        df = df.dropna(subset=required_fields).copy()
        if df.empty:
            raise QgsProcessingException(self.tr("No rows remain after checking coordinates, depth, and available predictors."))

        raw_depths = df[depth_col].to_numpy(dtype=float)
        has_negative = np.any(raw_depths < 0)
        df['positive_depth'] = -df[depth_col] if has_negative else df[depth_col]
        rows_before_depth_range = len(df)
        df = df[(df['positive_depth'] >= s_min) & (df['positive_depth'] <= s_max)].copy()
        if df.empty:
            raise QgsProcessingException(f"No control points remain in the positive depth range [{s_min} m, {s_max} m].")

        retained_before_tide = len(df)
        df = df[(df['positive_depth'] + training_tide) > 0].copy()
        if df.empty:
            raise QgsProcessingException(self.tr("Training tide correction leaves no points underwater at image time."))

        pixel_rows, pixel_cols, geotransform, grid_width, grid_height = self._pixel_indices(
            df, grid_layer, csv_crs, context
        )
        df['_pixel_row'] = pixel_rows
        df['_pixel_col'] = pixel_cols
        raw_sample_count = len(df)

        aggregate_columns = {feature: 'median' for feature in all_features}
        aggregate_columns['positive_depth'] = 'median'
        grouped = df.groupby(['_pixel_row', '_pixel_col'], sort=True)
        pixel_df = grouped.agg(aggregate_columns).reset_index()
        pixel_df['point_count'] = grouped.size().to_numpy(dtype=int)
        cell_cols = pixel_df['_pixel_col'].to_numpy(dtype=float) + 0.5
        cell_rows = pixel_df['_pixel_row'].to_numpy(dtype=float) + 0.5
        pixel_df['_pixel_x'] = geotransform[0] + cell_cols * geotransform[1] + cell_rows * geotransform[2]
        pixel_df['_pixel_y'] = geotransform[3] + cell_cols * geotransform[4] + cell_rows * geotransform[5]

        block_x = np.floor((pixel_df['_pixel_x'].to_numpy() - geotransform[0]) / block_size).astype(int)
        block_y = np.floor((geotransform[3] - pixel_df['_pixel_y'].to_numpy()) / block_size).astype(int)
        block_groups = np.asarray([f"{bx}:{by}" for bx, by in zip(block_x, block_y)], dtype=object)
        unique_blocks = np.unique(block_groups)
        if len(unique_blocks) < n_folds:
            raise QgsProcessingException(self.tr(f"Only {len(unique_blocks)} occupied spatial blocks; reduce block size or fold count."))

        y_mllw = pixel_df['positive_depth'].to_numpy(dtype=float)
        y_image_depth = y_mllw + training_tide
        if np.any(y_image_depth <= 0):
            raise QgsProcessingException(self.tr("Some pixel-median depths are non-positive at the supplied image-time water level."))
        y_log = np.log(y_image_depth + 0.1)
        base_splits = GroupKFold(n_splits=n_folds).split(
            np.zeros(len(pixel_df)), y_log, groups=block_groups
        )
        pixel_xy = pixel_df[['_pixel_x', '_pixel_y']].to_numpy(dtype=float)
        cv_splits = []
        for fold_number, (train_idx, test_idx) in enumerate(base_splits, start=1):
            if buffer_distance > 0:
                neighbor_index = NearestNeighbors(radius=buffer_distance, algorithm='kd_tree')
                neighbor_index.fit(pixel_xy[test_idx])
                near_test = neighbor_index.radius_neighbors(
                    pixel_xy[train_idx], return_distance=False
                )
                train_idx = train_idx[np.asarray([len(neighbors) == 0 for neighbors in near_test])]
            if len(train_idx) == 0:
                raise QgsProcessingException(self.tr(
                    f'Spatial buffer removed every training pixel for fold {fold_number}; reduce the buffer or block size.'
                ))
            cv_splits.append((train_idx, test_idx))
        max_depth = None if max_depth_param == 0 else max_depth_param
        model_parameters = {
            'n_estimators': 200,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

        if not visible_features:
            visible_features = selected_features
        matrices = {
            'visible_only': (visible_features, pixel_df[visible_features].to_numpy(dtype=float)),
            'all_available': (all_features, pixel_df[all_features].to_numpy(dtype=float))
        }
        cv_results = {}
        permutation_importance = {}
        for set_name, (features, matrix) in matrices.items():
            if set_name in cv_results:
                continue
            use_permutation = set_name == selected_set_name
            predictions, fold_ids, importance = self._run_rf_cv(
                matrix, y_log, y_mllw, training_tide, block_groups, cv_splits,
                model_parameters, features, calculate_permutation=use_permutation
            )
            cv_results[set_name] = {
                'features': features,
                'predictions': predictions,
                'fold_ids': fold_ids,
                'metrics': self._regression_metrics(y_mllw, predictions)
            }
            if use_permutation:
                permutation_importance = importance

        selected_result = cv_results[selected_set_name]
        selected_predictions = selected_result['predictions']
        depth_metrics = self._depth_metrics(y_mllw, selected_predictions, s_min, s_max, depth_bin_width)
        ablation_metrics = {name: result['metrics'] for name, result in cv_results.items()}

        baselines = {}
        if 'rhow_443' in pixel_df.columns and 'rhow_560' in pixel_df.columns:
            stumpf_reflectance = pixel_df[['rhow_443', 'rhow_560']].to_numpy(dtype=float)
            stumpf_pred = self._run_linear_baseline(
                'stumpf_ratio', stumpf_reflectance, y_image_depth, y_mllw,
                training_tide, cv_splits, stumpf_n=stumpf_n
            )
            baselines['stumpf_style_blue_green_ratio'] = self._regression_metrics(y_mllw, stumpf_pred)
        else:
            baselines['stumpf_style_blue_green_ratio'] = {'unavailable_reason': 'rhow_443 and rhow_560 are required'}

        lyzenga_names = [name for name in ['rhow_443', 'rhow_492', 'rhow_560', 'rhow_665'] if name in pixel_df.columns]
        if len(lyzenga_names) >= 2:
            lyzenga_reflectance = pixel_df[lyzenga_names].to_numpy(dtype=float)
            lyzenga_pred = self._run_linear_baseline(
                'lyzenga_log_linear', lyzenga_reflectance, y_image_depth, y_mllw,
                training_tide, cv_splits
            )
            baselines['lyzenga_style_log_linear'] = {
                **self._regression_metrics(y_mllw, lyzenga_pred),
                'bands': lyzenga_names
            }
        else:
            baselines['lyzenga_style_log_linear'] = {'unavailable_reason': 'at least two visible bands are required'}

        sensitivity = {}
        if run_sensitivity:
            selected_matrix = matrices[selected_set_name][1]
            for candidate_depth in (None, 10, 20):
                for candidate_leaf in (1, 3, 5):
                    for candidate_features in (1.0, 0.5):
                        params = {
                            'n_estimators': 200,
                            'max_depth': candidate_depth,
                            'min_samples_leaf': candidate_leaf,
                            'max_features': candidate_features
                        }
                        predictions, _, _ = self._run_rf_cv(
                            selected_matrix, y_log, y_mllw, training_tide,
                            block_groups, cv_splits, params,
                            matrices[selected_set_name][0], calculate_permutation=False
                        )
                        key = f"max_depth={candidate_depth}, min_samples_leaf={candidate_leaf}, max_features={candidate_features:g}"
                        sensitivity[key] = self._regression_metrics(y_mllw, predictions)

        rf = RandomForestRegressor(
            n_estimators=model_parameters['n_estimators'],
            max_depth=model_parameters['max_depth'],
            min_samples_leaf=model_parameters['min_samples_leaf'],
            max_features=model_parameters['max_features'],
            oob_score=False,
            random_state=42,
            n_jobs=-1
        )
        X_final = matrices[selected_set_name][1]
        rf.fit(X_final, y_log)
        selected_metrics = selected_result['metrics']
        full_rf_parameters = rf.get_params(deep=True)

        feedback.pushInfo(f"CSV rows: {input_row_count}; complete rows used: {valid_before_range}; excluded for invalid/missing coordinates, depth, or predictors: {input_row_count - valid_before_range}.")
        feedback.pushInfo(f"Rows outside configured depth range: {rows_before_depth_range - retained_before_tide}.")
        feedback.pushInfo(f"Pixel aggregation: {raw_sample_count} soundings -> {len(pixel_df)} raster pixels.")
        feedback.pushInfo(f"Dropped {retained_before_tide - raw_sample_count} rows that were exposed at the supplied image-time tide.")
        feedback.pushInfo(f"RF feature set: {selected_set_name}: {selected_result['features']}")
        feedback.pushInfo(f"Fitted Random Forest parameters: {full_rf_parameters}")
        feedback.pushInfo(f"Spatial-block CV: {n_folds} folds, {block_size:g} block size, {buffer_distance:g} buffer, raster CRS units, {len(unique_blocks)} occupied blocks.")
        if buffer_distance == 0:
            feedback.pushInfo(self.tr(
                'WARNING: No spatial buffer was applied. Blocks are held out, but neighboring training pixels can remain close to test pixels; do not describe these results as fully independent.'
            ))
        feedback.pushInfo(f"Training tide relative to MLLW: {training_tide:+.3f} m; CV predictions converted back to MLLW.")
        feedback.pushInfo(f"Spatial-block CV R²: {selected_metrics['r2'] if selected_metrics['r2'] is not None else 'undefined'}")
        feedback.pushInfo(f"Spatial-block CV RMSE: {selected_metrics['rmse']:.3f} m")
        feedback.pushInfo(f"Spatial-block CV MAE: {selected_metrics['mae']:.3f} m")
        feedback.pushInfo(f"Spatial-block CV bias (prediction - reference): {selected_metrics['bias']:+.3f} m")
        feedback.pushInfo("Depth-stratified spatial-CV metrics:")
        for depth_band, stats in depth_metrics.items():
            feedback.pushInfo(f"  {depth_band}: n={stats['count']}, RMSE={stats['rmse']}, MAE={stats['mae']}, bias={stats['bias']}")
        feedback.pushInfo("Same-fold baseline results:")
        for name, stats in baselines.items():
            feedback.pushInfo(f"  {name}: {stats}")
        feedback.pushInfo("Permutation importance is mean held-out RMSE increase; it is predictive, not causal:")
        feedback.pushInfo(f"  {permutation_importance}")
        if run_sensitivity:
            feedback.pushInfo("RF hyperparameter sensitivity is descriptive; it does not select the best settings:")
            for name, stats in sensitivity.items():
                feedback.pushInfo(f"  {name}: RMSE={stats['rmse']}, R²={stats['r2']}")

        out_dir = os.path.dirname(model_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            import sklearn
            model_payload = {
                'payload_version': 2,
                'model': rf,
                'features': selected_result['features'],
                'feature_set': selected_set_name,
                'target': 'positive depth at image-time water level; MLLW conversion uses prediction-time tide',
                'training_tide_mllw': training_tide,
                'rf_parameters': full_rf_parameters,
                'metrics': {
                    'validation_method': 'spatial_block_cv_no_buffer',
                    'primary': selected_metrics,
                    'depth_strata': depth_metrics,
                    'feature_set_comparison': ablation_metrics,
                    'baselines': baselines,
                    'permutation_importance_mean_rmse_increase': permutation_importance,
                    'rf_hyperparameter_sensitivity': sensitivity,
                    'raw_sample_count': raw_sample_count,
                    'pixel_count': len(pixel_df),
                    'sample_count': len(pixel_df),
                    'input_row_count': input_row_count,
                    'complete_row_count': valid_before_range,
                    'rows_in_depth_range': retained_before_tide,
                    'rows_after_tide_filter': raw_sample_count,
                    'rows_removed_as_exposed_at_image_time': retained_before_tide - raw_sample_count,
                    'depth_column': depth_col,
                    'depth_sign_convention': 'negative_elevation' if has_negative else 'positive_depth',
                    'depth_limits_m': (s_min, s_max),
                    'block_size': block_size,
                    'buffer_distance': buffer_distance,
                    'fold_count': n_folds,
                    'training_tide_mllw': training_tide,
                    'csv_coordinate_crs': csv_crs.authid(),
                    'grid_crs': grid_layer.crs().authid(),
                    'grid_geotransform': tuple(float(v) for v in geotransform),
                    'grid_width': grid_width,
                    'grid_height': grid_height,
                    'scikit_learn_version': sklearn.__version__,
                    'baseline_parameters': {
                        'stumpf_style': {
                            'scale_n': stumpf_n,
                            'bands': ['rhow_443', 'rhow_560'],
                            'calibration': 'ordinary least squares on positive image-time depth'
                        },
                        'lyzenga_style': {
                            'bands': lyzenga_names,
                            'training_fold_band_offset': 'fold-specific training minimum plus 1e-6',
                            'calibration': 'ordinary least squares on positive image-time depth'
                        }
                    }
                },
                'cv_splits': [
                    {
                        'fold': fold_number,
                        'train_pixel_indices': np.asarray(train_idx, dtype=int),
                        'test_pixel_indices': np.asarray(test_idx, dtype=int)
                    }
                    for fold_number, (train_idx, test_idx) in enumerate(cv_splits, start=1)
                ],
                'cv_predictions': {
                    'pixel_row': pixel_df['_pixel_row'].to_numpy(dtype=int),
                    'pixel_col': pixel_df['_pixel_col'].to_numpy(dtype=int),
                    'point_count': pixel_df['point_count'].to_numpy(dtype=int),
                    'observed_depth_mllw': y_mllw,
                    'predicted_depth_mllw': selected_predictions,
                    'predicted_image_depth': selected_predictions + training_tide,
                    'fold': selected_result['fold_ids'],
                    'block_id': block_groups,
                    'feature_set': selected_set_name
                }
            }
            joblib.dump(model_payload, model_path)
        except Exception as e:
            raise QgsProcessingException(f"Failed to save trained model payload: {str(e)}")

        return {self.MODEL_OUTPUT: model_path}
