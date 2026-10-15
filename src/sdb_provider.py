#Refactored & Optimized sdb_provider.py
#Tuesday 13 Oct 2026 3:45pm. Phil Standard time.

# -*- coding: utf-8 -*-
import os
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon 

from .sdb_trainer_alg import SDBTrainerAlgorithm
from .sdb_generator_alg import SDBGeneratorAlgorithm


class SDBProvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()

    def loadAlgorithms(self):
        """
        Registers the SDB algorithms into the QGIS Processing Toolbox.
        """
        self.addAlgorithm(SDBTrainerAlgorithm())
        self.addAlgorithm(SDBGeneratorAlgorithm())

    def id(self):
        """
        Unique identifier for the processing provider.
        """
        return 'sdb_suite'

    def name(self):
        """
        Human-readable name displayed in the QGIS Processing Toolbox header.
        """
        return 'SDB Production Suite'

    def icon(self):
        """
        Loads custom toolbar/toolbox icon if present in plugin directory.
        """
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()

    @staticmethod
    def get_sdb_interpretation(r2: float, rmse: float, bias: float, max_depth: float = 15.0) -> dict:
        """
        Calculates hydrographic precision labels and IHO S-44 suitability
        metrics for the SDB Production Suite training feedback logs.
        
        Parameters:
            r2 (float): Coefficient of determination from model cross-validation.
            rmse (float): Root Mean Square Error in meters.
            bias (float): Mean vertical error (predicted - actual) in meters.
            max_depth (float): Maximum shallow water depth evaluated (meters).
            
        Returns:
            dict: Formatted text strings for R2, RMSE, and Bias with hydrographic context.
        """
        # Protect against division by zero or invalid max depth
        safe_max_depth = max(float(max_depth), 0.1)

        # 1. R2 Interpretation (Model Goodness of Fit)
        if r2 >= 0.85:
            r2_desc = "Strong Correlation"
        elif r2 >= 0.70:
            r2_desc = "Moderate Correlation"
        elif r2 >= 0.50:
            r2_desc = "Weak Correlation (Review Training Soundings)"
        else:
            r2_desc = "Poor Correlation (High Noise / Turbidity Risk)"

        # 2. RMSE Interpretation (10% Depth Benchmark & IHO Vertical Uncertainty Context)
        error_ratio = (rmse / safe_max_depth) * 100.0
        if error_ratio < 5.0:
            rmse_desc = "Elite / Survey Grade (IHO Order 1b Compliant Candidate)"
        elif error_ratio < 10.0:
            rmse_desc = "High Precision (IHO Order 2 Compliant Candidate)"
        elif error_ratio < 20.0:
            rmse_desc = "Moderate / Acceptable (Reconnaissance Mapping)"
        else:
            rmse_desc = "Low Precision (Check ACOLITE Corrections / Turbidity Masks)"

        # 3. Bias Interpretation (Vertical Datum Alignment)
        if abs(bias) < 0.10:
            bias_desc = "Near-Perfect Vertical Alignment"
        elif bias > 0.10:
            bias_desc = f"Systemic Overestimation (+{bias:.2f}m Deep Bias)"
        else:
            bias_desc = f"Systemic Underestimation ({bias:.2f}m Shallow Bias)"

        return {
            "R2_Text": f"{r2:.3f} ({r2_desc})",
            "RMSE_Text": f"{rmse:.2f} m ({rmse_desc})",
            "Bias_Text": f"{bias:.2f} m ({bias_desc})"
        }