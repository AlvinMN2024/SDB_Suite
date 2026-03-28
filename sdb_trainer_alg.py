# -*- coding: utf-8 -*-
import os
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterFile, 
                       QgsProcessingParameterNumber, QgsProcessingParameterFileDestination,
                       QgsProcessingException)

import pandas as pd
import numpy as np

# --- Safe Dependency Imports ---
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score # Added for validation
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
            "<br><b>Aborlan Apurawan Edition</b>"
        )

    def group(self): return self.tr('SDB Suite')
    def groupId(self): return 'sdb_suite'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.CSV_INPUT, self.tr('Input CSV'), extension='csv'))
        self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MIN, self.tr('Min Depth (Positive for depth)'), defaultValue=0.5, type=QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MAX, self.tr('Max Depth (Positive for depth)'), defaultValue=20.0, type=QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterFileDestination(self.MODEL_OUTPUT, self.tr('Save Model (.pkl)'), 'Pickle (*.pkl)'))

    def processAlgorithm(self, parameters, context, feedback):
        if not HAS_SKLEARN:
            raise QgsProcessingException(self.tr("CRITICAL ERROR: 'scikit-learn' is not installed."))

        csv_path = self.parameterAsFile(parameters, self.CSV_INPUT, context)
        model_path = self.parameterAsFileOutput(parameters, self.MODEL_OUTPUT, context)
        s_min = self.parameterAsDouble(parameters, self.SHALLOW_MIN, context)
        s_max = self.parameterAsDouble(parameters, self.SHALLOW_MAX, context)

        df = pd.read_csv(csv_path).dropna()
        
        # Filtering data based on your MLLW column
        # Assuming MLLW is negative in your CSV (e.g., -5.0m)
        df = df[(df['MLLW'] <= -s_min) & (df['MLLW'] >= -s_max)]
        
        if df.empty:
            raise QgsProcessingException("No data points found in the specified depth range!")

        # Feature Engineering
        df['ln_665'] = np.log(df['rhow_665'] + 0.001)
        possible_feats = ['ln_665', 'rhow_443', 'rhow_492', 'rhow_560', 'TUR_Nechad']
        actual_feats = [f for f in possible_feats if f in df.columns]
        
        X = df[actual_feats]
        # We model the absolute depth (positive values)
        y_true = -df['MLLW'] 
        y_log = np.log(y_true + 0.1)

        # Training
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y_log)
        
        # Validation Calculations
        y_pred_log = rf.predict(X)
        y_pred = np.exp(y_pred_log) - 0.1
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = np.mean(y_pred - y_true)

        # Import the Provider to use the Interpretation logic
        from .sdb_provider import SDBProvider
        stats = SDBProvider.get_sdb_interpretation(r2, rmse, bias, max_depth=s_max)

        # Final Feedback Report
        feedback.pushInfo("-" * 40)
        feedback.pushInfo(f"SUCCESS: Model Trained for {len(df)} points.")
        feedback.pushInfo(f"R2 Score: {stats['R2_Text']}")
        feedback.pushInfo(f"RMSE: {stats['RMSE_Text']}")
        feedback.pushInfo(f"Bias: {stats['Bias_Text']}")
        feedback.pushInfo("-" * 40)

        # Save model
        model_data = {'model': rf, 'features': actual_feats}
        joblib.dump(model_data, model_path)
        
        return {self.MODEL_OUTPUT: model_path}
