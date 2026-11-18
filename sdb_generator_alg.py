# -*- coding: utf-8 -*-
"""Apply an SDB Production Suite Random Forest payload to a raster stack."""
import os

import numpy as np
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterBand,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException,
)

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


class SDBGeneratorAlgorithm(QgsProcessingAlgorithm):
    INPUT_STACK = 'INPUT_STACK'
    MODEL_INPUT = 'MODEL_INPUT'
    WATER_MASK = 'WATER_MASK'
    PREDICTION_TIDE = 'PREDICTION_TIDE'

    BAND_443 = 'BAND_443'
    BAND_492 = 'BAND_492'
    BAND_560 = 'BAND_560'
    BAND_665 = 'BAND_665'
    BAND_TUR = 'BAND_TUR'
    BAND_SPM = 'BAND_SPM'
    BAND_704 = 'BAND_704'
    BAND_833 = 'BAND_833'
    BAND_865 = 'BAND_865'

    DEPTH_STRETCH = 'DEPTH_STRETCH'
    BIAS_CORRECTION = 'BIAS_CORRECTION'
    MAX_DEPTH_CLAMP = 'MAX_DEPTH_CLAMP'
    OUTPUT_RASTER = 'OUTPUT_RASTER'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return SDBGeneratorAlgorithm()

    def name(self):
        return 'sdb_generator'

    def displayName(self):
        return self.tr('2. Generate Bathymetry Map (RF)')

    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        return QIcon(icon_path) if os.path.exists(icon_path) else super().icon()

    def shortHelpString(self):
        return self.tr(
            '<h2>SDB Production Suite: Map Generator</h2>'
            '<p>Applies a saved Random Forest model to a raster stack. The selected band numbers must '
            'match the bands used to create the training CSV.</p>'
            '<p>The model predicts positive depth at the image-time water level. Supply that water level '
            'relative to MLLW (positive above MLLW) to convert output to negative MLLW elevations. '
            'The default of 0 m applies no tide correction.</p>'
            '<p>An optional water mask must be aligned to the input stack; values greater than zero are water. '
            'Without a mask, the algorithm masks nodata and non-finite predictor pixels only. It does not '
            'infer land from a fixed reflectance threshold.</p>'
            '<p>Depth stretch and bias are post-processing adjustments and are not part of the reported '
            'cross-validation metrics. Keep them at 1 and 0 unless independently justified.</p>'
        )

    def group(self):
        return self.tr('SDB Suite')

    def groupId(self):
        return 'sdb_suite'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_STACK, self.tr('Satellite reflectance / predictor raster stack')
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.MODEL_INPUT, self.tr('Trained model payload (.pkl)'), extension='pkl'
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.WATER_MASK, self.tr('Optional aligned water mask (water values > 0)'), optional=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.PREDICTION_TIDE,
            self.tr('Image-time water level relative to MLLW (m; positive above MLLW)'),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.0
        ))

        band_parameters = (
            (self.BAND_443, '443 nm / coastal blue', 1),
            (self.BAND_492, '492 nm / blue', 2),
            (self.BAND_560, '560 nm / green', 3),
            (self.BAND_665, '665 nm / red', 4),
            (self.BAND_TUR, 'TUR predictor', 5),
            (self.BAND_SPM, 'SPM predictor', 6),
            (self.BAND_704, '704 nm / red edge', 7),
            (self.BAND_833, 'NIR predictor (legacy 833/883 label)', 8),
            (self.BAND_865, '865 nm / NIR', 9),
        )
        for parameter_name, label, default_index in band_parameters:
            self.addParameter(QgsProcessingParameterBand(
                parameter_name,
                self.tr(f'Band index: {label}'),
                defaultValue=default_index,
                parentLayerParameterName=self.INPUT_STACK
            ))

        self.addParameter(QgsProcessingParameterNumber(
            self.DEPTH_STRETCH,
            self.tr('Post-processing depth stretch (1 = unchanged)'),
            defaultValue=1.0,
            minValue=0.001,
            type=QgsProcessingParameterNumber.Type.Double
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.BIAS_CORRECTION,
            self.tr('Post-processing depth offset (m; added to positive depth)'),
            defaultValue=0.0,
            type=QgsProcessingParameterNumber.Type.Double
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_DEPTH_CLAMP,
            self.tr('Maximum output depth to retain (m; 0 disables this mask)'),
            defaultValue=20.0,
            minValue=0.0,
            type=QgsProcessingParameterNumber.Type.Double
        ))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_RASTER, self.tr('Output negative MLLW elevation raster (.tif)')
        ))

    @staticmethod
    def _same_grid(first, second):
        first_gt = first.GetGeoTransform()
        second_gt = second.GetGeoTransform()
        if first.RasterXSize != second.RasterXSize or first.RasterYSize != second.RasterYSize:
            return False
        if any(abs(a - b) > 1e-8 for a, b in zip(first_gt, second_gt)):
            return False
        return first.GetProjection() == second.GetProjection()

    def processAlgorithm(self, parameters, context, feedback):
        if not HAS_JOBLIB:
            raise QgsProcessingException(self.tr('joblib is required in the QGIS Python environment.'))
        if not HAS_GDAL:
            raise QgsProcessingException(self.tr('GDAL Python bindings are required in the QGIS Python environment.'))

        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STACK, context)
        model_path = self.parameterAsFile(parameters, self.MODEL_INPUT, context)
        water_layer = self.parameterAsRasterLayer(parameters, self.WATER_MASK, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        prediction_tide = self.parameterAsDouble(parameters, self.PREDICTION_TIDE, context)
        stretch = self.parameterAsDouble(parameters, self.DEPTH_STRETCH, context)
        bias = self.parameterAsDouble(parameters, self.BIAS_CORRECTION, context)
        max_depth = self.parameterAsDouble(parameters, self.MAX_DEPTH_CLAMP, context)

        if raster_layer is None or not raster_layer.isValid():
            raise QgsProcessingException(self.tr('Select a valid predictor raster stack.'))
        if water_layer is not None and not water_layer.isValid():
            raise QgsProcessingException(self.tr('Selected water mask is not a valid raster layer.'))
        if stretch <= 0 or max_depth < 0:
            raise QgsProcessingException(self.tr('Depth stretch must be positive and maximum depth cannot be negative.'))

        try:
            payload = joblib.load(model_path)
        except Exception as e:
            raise QgsProcessingException(f'Could not load the model file: {e}')

        if isinstance(payload, dict):
            model = payload.get('model')
            model_features = payload.get('features')
            training_tide = payload.get('training_tide_mllw')
        else:
            model = payload
            model_features = None
            training_tide = None
        if model is None or not hasattr(model, 'predict'):
            raise QgsProcessingException(self.tr('The model file does not contain a usable estimator.'))

        band_indices = {
            'rhow_443': self.parameterAsInt(parameters, self.BAND_443, context),
            'rhow_492': self.parameterAsInt(parameters, self.BAND_492, context),
            'rhow_560': self.parameterAsInt(parameters, self.BAND_560, context),
            'rhow_665': self.parameterAsInt(parameters, self.BAND_665, context),
            'TUR_Nechad': self.parameterAsInt(parameters, self.BAND_TUR, context),
            'TUR_Nechad2016_665': self.parameterAsInt(parameters, self.BAND_TUR, context),
            'SPM_Nechad': self.parameterAsInt(parameters, self.BAND_SPM, context),
            'SPM_Nechad2016_665': self.parameterAsInt(parameters, self.BAND_SPM, context),
            'rhow_704': self.parameterAsInt(parameters, self.BAND_704, context),
            'rhow_833': self.parameterAsInt(parameters, self.BAND_833, context),
            'rhow_883': self.parameterAsInt(parameters, self.BAND_833, context),
            'rhow_865': self.parameterAsInt(parameters, self.BAND_865, context),
        }
        if model_features is None:
            # Compatibility for older saved estimators that did not include feature names.
            model_features = [
                'ln_665', 'rhow_443', 'rhow_492', 'rhow_560', 'rhow_665',
                'TUR_Nechad', 'SPM_Nechad', 'rhow_704', 'rhow_833', 'rhow_865'
            ]
            feedback.pushInfo(self.tr(
                'WARNING: This older model has no saved feature names. Applying the legacy 10-column feature order; '
                'confirm it matches the original training setup.'
            ))
        else:
            model_features = list(model_features)
        if not model_features or len(set(model_features)) != len(model_features):
            raise QgsProcessingException(self.tr('Model feature list is empty or contains duplicates.'))
        unknown = [name for name in model_features if name != 'ln_665' and name not in band_indices]
        if unknown:
            raise QgsProcessingException(self.tr(f'Model contains unsupported feature names: {unknown}'))
        expected_features = getattr(model, 'n_features_in_', None)
        if expected_features is not None and int(expected_features) != len(model_features):
            raise QgsProcessingException(self.tr(
                f'Model expects {expected_features} features, but its payload names {len(model_features)}.'
            ))

        ds = gdal.Open(raster_layer.dataProvider().dataSourceUri())
        if ds is None:
            raise QgsProcessingException(self.tr('Could not open the input raster stack through GDAL.'))
        cols, rows = ds.RasterXSize, ds.RasterYSize
        raster_band_count = ds.RasterCount
        needed_indices = {band_indices[name] for name in model_features if name != 'ln_665'}
        if 'ln_665' in model_features:
            needed_indices.add(band_indices['rhow_665'])
        for band_index in needed_indices:
            if band_index < 1 or band_index > raster_band_count:
                ds = None
                raise QgsProcessingException(self.tr(
                    f'Model feature band index {band_index} is outside the raster band range 1..{raster_band_count}.'
                ))

        mask_ds = None
        if water_layer is not None and water_layer.isValid():
            mask_ds = gdal.Open(water_layer.dataProvider().dataSourceUri())
            if mask_ds is None:
                ds = None
                raise QgsProcessingException(self.tr('Could not open the water mask through GDAL.'))
            if not self._same_grid(ds, mask_ds):
                mask_ds = None
                ds = None
                raise QgsProcessingException(self.tr(
                    'Water mask grid, dimensions, and CRS must exactly match the predictor stack. Align it first.'
                ))

        feedback.pushInfo(f'Raster dimensions: {cols} × {rows}; model predictors: {model_features}')
        if training_tide is None:
            feedback.pushInfo(self.tr(
                'WARNING: Model payload has no training tide metadata. Prediction tide is applied as supplied; '
                'the model may not be on an MLLW-compatible depth basis.'
            ))
        else:
            feedback.pushInfo(
                f'Model training tide: {float(training_tide):+.3f} m relative to MLLW; '
                f'prediction tide: {prediction_tide:+.3f} m.'
            )
        if stretch != 1.0 or bias != 0.0:
            feedback.pushInfo(self.tr(
                'WARNING: Stretch/offset are post-processing changes and are not covered by the saved CV metrics.'
            ))

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
        if out_ds is None:
            mask_ds = None
            ds = None
            raise QgsProcessingException(self.tr('Could not create the output raster.'))
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        nodata = -9999.0
        out_band.SetNoDataValue(nodata)

        block_size = 512
        total_blocks = int(np.ceil(cols / block_size) * np.ceil(rows / block_size))
        current_block = 0
        band_handles = {index: ds.GetRasterBand(index) for index in needed_indices}
        band_nodata = {index: band.GetNoDataValue() for index, band in band_handles.items()}
        mask_band = mask_ds.GetRasterBand(1) if mask_ds is not None else None
        mask_nodata = mask_band.GetNoDataValue() if mask_band is not None else None

        try:
            for y in range(0, rows, block_size):
                y_size = min(block_size, rows - y)
                for x in range(0, cols, block_size):
                    x_size = min(block_size, cols - x)
                    arrays = {}
                    valid = np.ones((y_size, x_size), dtype=bool)
                    for index, band in band_handles.items():
                        values = band.ReadAsArray(x, y, x_size, y_size).astype(np.float32)
                        arrays[index] = values
                        valid &= np.isfinite(values)
                        no_data = band_nodata[index]
                        if no_data is not None:
                            if np.isnan(no_data):
                                valid &= ~np.isnan(values)
                            else:
                                valid &= values != no_data

                    if mask_band is not None:
                        mask_values = mask_band.ReadAsArray(x, y, x_size, y_size)
                        valid &= np.isfinite(mask_values) & (mask_values > 0)
                        if mask_nodata is not None and not np.isnan(mask_nodata):
                            valid &= mask_values != mask_nodata

                    output = np.full((y_size, x_size), nodata, dtype=np.float32)
                    if np.any(valid):
                        feature_arrays = {}
                        for feature_name, band_index in band_indices.items():
                            if band_index in arrays:
                                feature_arrays[feature_name] = arrays[band_index]
                        if 'rhow_665' in feature_arrays:
                            feature_arrays['ln_665'] = np.log(np.maximum(feature_arrays['rhow_665'], 1e-6))
                        X = np.column_stack([
                            feature_arrays[name][valid] for name in model_features
                        ]).astype(np.float32, copy=False)
                        log_depth = np.asarray(model.predict(X), dtype=float)
                        if len(log_depth) != int(np.count_nonzero(valid)) or not np.all(np.isfinite(log_depth)):
                            raise QgsProcessingException(self.tr('Model returned invalid predictions.'))
                        positive_depth_mllw = np.exp(log_depth) - 0.1 - prediction_tide
                        positive_depth_mllw = positive_depth_mllw * stretch + bias
                        if not np.all(np.isfinite(positive_depth_mllw)):
                            raise QgsProcessingException(self.tr('Model produced non-finite depth values.'))
                        depth = -positive_depth_mllw
                        if max_depth > 0:
                            depth[positive_depth_mllw > max_depth] = nodata
                        output[valid] = depth.astype(np.float32)
                    out_band.WriteArray(output, x, y)
                    current_block += 1
                    feedback.setProgress(int((current_block / total_blocks) * 100))
                    if feedback.isCanceled():
                        raise QgsProcessingException(self.tr('Processing canceled.'))
        except Exception:
            out_band = None
            out_ds = None
            mask_ds = None
            ds = None
            raise

        out_band.FlushCache()
        out_band = None
        out_ds = None
        mask_ds = None
        ds = None
        feedback.pushInfo(f'Output written: {output_path}')
        return {self.OUTPUT_RASTER: output_path}
