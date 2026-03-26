# -*- coding: utf-8 -*-
import os
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterFile, 
                       QgsProcessingParameterNumber, QgsProcessingParameterFileDestination,
                       QgsProcessingException) # Added Exception for errors

import pandas as pd
import numpy as np

# --- Safe Dependency Imports ---
try:
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class SDBTrainerAlgorithm(QgsProcessingAlgorithm):
    CSV_INPUT = 'CSV_INPUT'
    SHALLOW_MIN = 'SHALLOW_MIN'
    SHALLOW_MAX = 'SHALLOW_MAX'
    MODEL_OUTPUT = 'MODEL_OUTPUT'

    def tr(self, string): return QCoreApplication.translate('Processing', string)
    def createInstance(self): return SDBTrainerAlgorithm()
    def name(self): return 'sdb_trainer'
    def displayName(self): return self.tr('1. Train Bathymetry Model (RF)')
    
    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        return QIcon(icon_path) if os.path.exists(icon_path) else super().icon()

    def shortHelpString(self):
        return self.tr(
            "<h2>SDB Production Suite: Model Trainer</h2>"
            "<p>Trains a Random Forest (RF) model to predict water depth from satellite reflectance.</p>"
            "<h3>Pro-Tip:</h3>"
            "<p>Ensure your CSV excludes land pixels.</p>"
            "<br><b>Apurawan Aborlan Edition</b>"
        )

    def group(self): return self.tr('SDB Suite')
    def groupId(self): return 'sdb_suite'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.CSV_INPUT, self.tr('Input CSV'), extension='csv'))
        self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MIN, self.tr('Min Depth'), defaultValue=-20.0, type=QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MAX, self.tr('Max Depth'), defaultValue=-1.0, type=QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterFileDestination(self.MODEL_OUTPUT, self.tr('Save Model (.pkl)'), 'Pickle (*.pkl)'))

    def processAlgorithm(self, parameters, context, feedback):
        # 1. Dependency Check First
        if not HAS_SKLEARN:
            raise QgsProcessingException(
                self.tr("CRITICAL ERROR: 'scikit-learn' is not installed. "
                        "Please run 'sudo apt install python3-sklearn' (Linux) "
                        "or 'pip install scikit-learn' (Windows OSGeo4W Shell) to use this tool.")
            )

        csv_path = self.parameterAsFile(parameters, self.CSV_INPUT, context)
        model_path = self.parameterAsFileOutput(parameters, self.MODEL_OUTPUT, context)
        
        df = pd.read_csv(csv_path).dropna()
        s_min = self.parameterAsDouble(parameters, self.SHALLOW_MIN, context)
        s_max = self.parameterAsDouble(parameters, self.SHALLOW_MAX, context)
        
        df = df[(df['MLLW'] > s_min) & (df['MLLW'] < s_max)]
        
        if 'rhow_665' not in df.columns:
            feedback.reportError("CRITICAL ERROR: Column 'rhow_665' missing!")
            return {}
        
        df['ln_665'] = np.log(df['rhow_665'] + 0.001)
        possible_feats = ['ln_665', 'rhow_443', 'rhow_492', 'rhow_560', 'TUR_Nechad', 'SPM_Nechad']
        actual_feats = [f for f in possible_feats if f in df.columns]
        
        X = df[actual_feats]
        y = np.log(-df['MLLW'] + 0.1)

        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        model_data = {'model': rf, 'features': actual_feats}
        joblib.dump(model_data, model_path)
        
        feedback.pushInfo(f"Model trained. R2 Score: {rf.score(X, y):.4f}")
        return {self.MODEL_OUTPUT: model_path}