#Refactored & Optimized sdb_generator_alg.py
#Tuesday 13 Oct 2026 3:30pm. Phil Standard time.

# -*- coding: utf-8 -*-
import os
import numpy as np
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)

# --- Safe Dependency Imports ---
try:
    import joblib
    import rasterio
    from rasterio.windows import Window
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class SDBGeneratorAlgorithm(QgsProcessingAlgorithm):
    INPUT_TIF = 'INPUT_TIF'
    MODEL = 'MODEL'
    B443, B492, B560, B665, BTUR, BSPM = 'B443', 'B492', 'B560', 'B665', 'BTUR', 'BSPM'
    BIAS = 'BIAS'
    STRETCH = 'STRETCH'
    OUTPUT = 'OUTPUT'

    def tr(self, string): 
        return QCoreApplication.translate('Processing', string)
        
    def createInstance(self): 
        return SDBGeneratorAlgorithm()
        
    def name(self): 
        return 'sdb_generator'
        
    def displayName(self): 
        return self.tr('2. Generate Bathymetry Map')
    
    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        return QIcon(icon_path) if os.path.exists(icon_path) else super().icon()

    def group(self): 
        return self.tr('SDB Suite')
        
    def groupId(self): 
        return 'sdb_suite'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_TIF, self.tr('Satellite Multi-band TIF')))
        self.addParameter(QgsProcessingParameterFile(self.MODEL, self.tr('Model File (.pkl)'), extension='pkl'))
        self.addParameter(QgsProcessingParameterNumber(self.B443, 'Band Index: 443nm', defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.B492, 'Band Index: 492nm', defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.B560, 'Band Index: 560nm', defaultValue=2))
        self.addParameter(QgsProcessingParameterNumber(self.B665, 'Band Index: 665nm', defaultValue=3))
        self.addParameter(QgsProcessingParameterNumber(self.BTUR, 'Band Index: TUR (Turbidity)', defaultValue=4))
        self.addParameter(QgsProcessingParameterNumber(self.BSPM, 'Band Index: SPM (Suspended Matter)', defaultValue=5))
        self.addParameter(QgsProcessingParameterNumber(self.STRETCH, 'Depth Stretch Constant', defaultValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(self.BIAS, 'Bias Correction (m)', defaultValue=-0.14))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, self.tr('Final Bathymetry Raster')))

    def processAlgorithm(self, parameters, context, feedback):
        if not HAS_DEPS:
            raise QgsProcessingException(
                self.tr("CRITICAL ERROR: Missing libraries! Please install 'scikit-learn', 'joblib', and 'rasterio' "
                        "in your QGIS Python environment.")
            )

        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        tif_layer = self.parameterAsRasterLayer(parameters, self.INPUT_TIF, context)
        tif_path = tif_layer.source()
        model_file = self.parameterAsFile(parameters, self.MODEL, context)
        
        stretch = self.parameterAsDouble(parameters, self.STRETCH, context)
        bias = self.parameterAsDouble(parameters, self.BIAS, context)
        
        # Load pre-trained model and metadata
        loaded_data = joblib.load(model_file)
        model = loaded_data['model']
        feats = loaded_data['features']

        band_map = {
            'rhow_443': parameters[self.B443],
            'rhow_492': parameters[self.B492],
            'rhow_560': parameters[self.B560],
            'rhow_665': parameters[self.B665],
            'TUR_Nechad': parameters[self.BTUR],
            'SPM_Nechad': parameters[self.BSPM],
            'ln_665': parameters[self.B665]
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(tif_path) as src:
            out_meta = src.profile.copy()
            out_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
            
            # Setup block-by-block windowed processing to keep memory usage under 100MB
            windows = [window for _, window in src.block_windows(1)]
            total_windows = len(windows)
            
            with rasterio.open(out_path, 'w', **out_meta) as dst:
                for idx_win, window in enumerate(windows):
                    # Check if user pressed Cancel in QGIS interface
                    if feedback.isCanceled():
                        return {self.OUTPUT: None}

                    data_stack = []
                    for f in feats:
                        band_idx = band_map.get(f, parameters[self.B665])
                        raw_data = src.read(int(band_idx), window=window).astype(np.float32)
                        
                        # Prevent log(0) or log of negative reflectance values
                        raw_cleaned = np.maximum(raw_data, 1e-6)
                        if 'ln_' in f or f == 'ln_665':
                            data_stack.append(np.log(raw_cleaned))
                        else:
                            data_stack.append(raw_cleaned)

                    # Stack features along last axis and flatten window pixels
                    stacked = np.stack(data_stack, axis=-1)
                    h, w, num_feats = stacked.shape
                    flat_input = stacked.reshape(-1, num_feats)

                    # Mask NaN / Inf values prior to model inference
                    valid_mask = ~np.isnan(flat_input).any(axis=1) & ~np.isinf(flat_input).any(axis=1)
                    predicted_flat = np.full(flat_input.shape[0], np.nan, dtype=np.float32)

                    if np.any(valid_mask):
                        raw_pred = model.predict(flat_input[valid_mask])
                        # Invert log-depth transform and apply stretch & bias
                        depth_raw = -np.exp(raw_pred)
                        depth_final = (depth_raw * stretch) + bias
                        
                        # Apply depth validity boundary (-60m to +2m)
                        depth_final[(depth_final > 2.0) | (depth_final < -60.0)] = np.nan
                        predicted_flat[valid_mask] = depth_final

                    # Reshape prediction back to 2D block window and write to output
                    depth_block = predicted_flat.reshape(h, w)
                    dst.write(depth_block.astype(np.float32), 1, window=window)

                    # Update QGIS Progress Bar
                    progress = float(idx_win + 1) / total_windows * 100.0
                    feedback.setProgress(int(progress))

        feedback.pushInfo(f"SDB Bathymetry generation complete. Stretch: {stretch}, Bias Applied: {bias}m")
        return {self.OUTPUT: out_path}