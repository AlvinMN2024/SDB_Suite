#Refactored & Optimized sdb_trainer_alg.py
#Tuesday 13 Oct 2026 3:35pm. Phil Standard time

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
    QgsProcessingException
)

# --- Safe Dependency Imports ---
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SDBTrainerAlgorithm(QgsProcessingAlgorithm):
    CSV_INPUT = 'CSV_INPUT'
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
            "  <li>Provide a CSV containing surface reflectances (rhow_*) and target depths.</li>"
            "  <li>Supports 'MLLW', 'depth', or 'Z' column names.</li>"
            "  <li>Out-of-Bag (OOB) cross-validation metrics are computed automatically.</li>"
            "</ul>"
            "<br><b>Aborlan Apurawan Edition</b>"
        )

    def group(self): 
        return self.tr('SDB Suite')
        
    def groupId(self): 
        return 'sdb_suite'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.CSV_INPUT, self.tr('Input Control Soundings CSV'), extension='csv'))
        self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MIN, self.tr('Min Depth (m, positive)'), defaultValue=0.5, type=QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterNumber(self.SHALLOW_MAX, self.tr('Max Depth (m, positive)'), defaultValue=20.0, type=QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterFileDestination(self.MODEL_OUTPUT, self.tr('Save Model (.pkl)'), 'Pickle (*.pkl)'))

    def processAlgorithm(self, parameters, context, feedback):
        if not HAS_SKLEARN:
            raise QgsProcessingException(self.tr("CRITICAL ERROR: 'scikit-learn' or 'joblib' is not installed in the QGIS Python environment."))

        csv_path = self.parameterAsFile(parameters, self.CSV_INPUT, context)
        model_path = self.parameterAsFileOutput(parameters, self.MODEL_OUTPUT, context)
        s_min = self.parameterAsDouble(parameters, self.SHALLOW_MIN, context)
        s_max = self.parameterAsDouble(parameters, self.SHALLOW_MAX, context)

        # 1. Read CSV and clean missing values
        try:
            df = pd.read_csv(csv_path).dropna()
        except Exception as e:
            raise QgsProcessingException(f"Failed to read CSV file: {str(e)}")

        # 2. Identify depth column dynamically
        depth_col = None
        for col in ['MLLW', 'mllw', 'depth', 'Depth', 'Z', 'z']:
            if col in df.columns:
                depth_col = col
                break

        if not depth_col:
            raise QgsProcessingException("CRITICAL ERROR: Could not find a valid depth column ('MLLW', 'depth', or 'Z') in CSV.")

        # Convert to positive bathymetric depth values
        raw_depths = df[depth_col].values
        if np.mean(raw_depths) < 0:
            df['positive_depth'] = -df[depth_col]
        else:
            df['positive_depth'] = df[depth_col]

        # Filter by depth range bounds
        df = df[(df['positive_depth'] >= s_min) & (df['positive_depth'] <= s_max)]
        
        if df.empty:
            raise QgsProcessingException(f"No control points remaining within the depth range [{s_min}m - {s_max}m].")

        # 3. Feature Engineering
        if 'rhow_665' in df.columns and 'ln_665' not in df.columns:
            df['ln_665'] = np.log(np.maximum(df['rhow_665'], 1e-6))

        candidate_feats = ['ln_665', 'rhow_443', 'rhow_492', 'rhow_560', 'rhow_665', 'TUR_Nechad', 'SPM_Nechad']
        actual_feats = [f for f in candidate_feats if f in df.columns]

        if not actual_feats:
            raise QgsProcessingException("CRITICAL ERROR: No valid spectral or turbidity features found in CSV.")

        X = df[actual_feats].values
        y_true = df['positive_depth'].values
        y_log = np.log(y_true + 0.1)

        # 4. Model Training with Out-of-Bag (OOB) validation
        feedback.pushInfo(f"Training Random Forest model using features: {actual_feats}")
        rf = RandomForestRegressor(
            n_estimators=200, 
            oob_score=True, 
            random_state=42, 
            n_jobs=-1
        )
        rf.fit(X, y_log)

        # 5. Out-Of-Bag (OOB) Validation Metrics
        oob_pred_log = rf.oob_prediction_
        y_pred_oob = np.exp(oob_pred_log) - 0.1

        r2_oob = r2_score(y_true, y_pred_oob)
        rmse_oob = np.sqrt(mean_squared_error(y_true, y_pred_oob))
        mae_oob = mean_absolute_error(y_true, y_pred_oob)
        bias_oob = np.mean(y_pred_oob - y_true)

        # Attempt to format metrics via provider helper if present
        try:
            from .sdb_provider import SDBProvider
            stats = SDBProvider.get_sdb_interpretation(r2_oob, rmse_oob, bias_oob, max_depth=s_max)
            r2_str = stats.get('R2_Text', f"{r2_oob:.3f}")
            rmse_str = stats.get('RMSE_Text', f"{rmse_oob:.2f} m")
            bias_str = stats.get('Bias_Text', f"{bias_oob:.2f} m")
        except Exception:
            r2_str = f"{r2_oob:.3f}"
            rmse_str = f"{rmse_oob:.2f} m"
            bias_str = f"{bias_oob:.2f} m"

        # 6. Report Hydrographic Validation Metrics
        feedback.pushInfo("=" * 50)
        feedback.pushInfo(f"SUCCESS: Model Trained on {len(df)} control points.")
        feedback.pushInfo(f"OOB R² Score    : {r2_str}")
        feedback.pushInfo(f"OOB RMSE        : {rmse_str}")
        feedback.pushInfo(f"OOB MAE         : {mae_oob:.2f} m")
        feedback.pushInfo(f"OOB Mean Bias   : {bias_str}")
        feedback.pushInfo("=" * 50)

        # 7. Save Model Artifact
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_payload = {
            'model': rf, 
            'features': actual_feats,
            'metrics': {
                'r2_oob': r2_oob,
                'rmse_oob': rmse_oob,
                'mae_oob': mae_oob,
                'bias_oob': bias_oob,
                'sample_count': len(df)
            }
        }
        joblib.dump(model_payload, model_path)

        return {self.MODEL_OUTPUT: model_path}