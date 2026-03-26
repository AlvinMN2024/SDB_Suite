# -*- coding: utf-8 -*-
import os
import numpy as np
import joblib
import rasterio
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon # Qt6 Requirement
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFile, QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterDestination)

class SDBGeneratorAlgorithm(QgsProcessingAlgorithm):
    INPUT_TIF = 'INPUT_TIF'
    MODEL = 'MODEL'
    B443, B492, B560, B665, BTUR, BSPM = 'B443','B492','B560','B665','BTUR','BSPM'
    BIAS = 'BIAS'
    STRETCH = 'STRETCH'
    OUTPUT = 'OUTPUT'

    def tr(self, string): return QCoreApplication.translate('Processing', string)
    def createInstance(self): return SDBGeneratorAlgorithm()
    def name(self): return 'sdb_generator'
    def displayName(self): return self.tr('2. Generate Bathymetry Map (Force-Run)')
    
    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        return QIcon(icon_path) if os.path.exists(icon_path) else super().icon()

    def group(self): return self.tr('SDB Suite')
    def groupId(self): return 'sdb_suite'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_TIF, self.tr('Satellite Multi-band TIF')))
        self.addParameter(QgsProcessingParameterFile(self.MODEL, self.tr('Model File (.pkl)'), extension='pkl'))
        self.addParameter(QgsProcessingParameterNumber(self.B443, 'Index: 443', defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.B492, 'Index: 492', defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.B560, 'Index: 560', defaultValue=2))
        self.addParameter(QgsProcessingParameterNumber(self.B665, 'Index: 665', defaultValue=3))
        self.addParameter(QgsProcessingParameterNumber(self.BTUR, 'Index: TUR', defaultValue=4))
        self.addParameter(QgsProcessingParameterNumber(self.BSPM, 'Index: SPM', defaultValue=5))
        self.addParameter(QgsProcessingParameterNumber(self.STRETCH, 'Depth Stretch', defaultValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(self.BIAS, 'Bias Correction (m)', defaultValue=-0.14))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, self.tr('Final Bathymetry Raster')))

    def processAlgorithm(self, parameters, context, feedback):
        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        tif_layer = self.parameterAsRasterLayer(parameters, self.INPUT_TIF, context)
        tif_path = tif_layer.source()
        model_file = self.parameterAsFile(parameters, self.MODEL, context)
        
        stretch = self.parameterAsDouble(parameters, self.STRETCH, context)
        bias = self.parameterAsDouble(parameters, self.BIAS, context)
        
        loaded_data = joblib.load(model_file)
        model = loaded_data['model']
        feats = loaded_data['features']

        with rasterio.open(tif_path) as src:
            band_map = {
                'rhow_443': parameters[self.B443], 'rhow_492': parameters[self.B492],
                'rhow_560': parameters[self.B560], 'rhow_665': parameters[self.B665],
                'TUR_Nechad': parameters[self.BTUR], 'SPM_Nechad': parameters[self.BSPM]
            }

            data_stack = []
            for f in feats:
                idx = parameters[self.B665] if f == 'ln_665' else band_map.get(f)
                raw_data = src.read(int(idx)).astype(float)
                raw_cleaned = np.maximum(raw_data, 1e-6)
                data_stack.append(np.log(raw_cleaned) if f == 'ln_665' else raw_cleaned)

            flat_input = np.stack(data_stack, axis=-1).reshape(-1, len(feats))
            prediction = model.predict(flat_input)
            
            depth_raw = -np.exp(prediction).reshape(src.shape)
            depth_final = (depth_raw * stretch) + bias
            depth_final[(depth_final > 2) | (depth_final < -60)] = np.nan

            out_meta = src.profile.copy()
            out_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with rasterio.open(out_path, 'w', **out_meta) as dst:
                dst.write(depth_final.astype(np.float32), 1)

        feedback.pushInfo(f"Bias Applied: {bias}m")
        return {self.OUTPUT: out_path}