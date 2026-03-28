# -*- coding: utf-8 -*-
import os
from qgis.core import QgsProcessingProvider
from PyQt6.QtGui import QIcon 
from .sdb_trainer_alg import SDBTrainerAlgorithm
from .sdb_generator_alg import SDBGeneratorAlgorithm

class SDBProvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()

    def loadAlgorithms(self):
        self.addAlgorithm(SDBTrainerAlgorithm())
        self.addAlgorithm(SDBGeneratorAlgorithm())

    def id(self):
        return 'sdb_suite'

    def name(self):
        return 'SDB Production Suite'

    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()

    @staticmethod
    def get_sdb_interpretation(r2, rmse, bias, max_depth=15.0):
        """
        Calculates hydrographic precision labels for the SDB Manual standards.
        """
        # 1. R2 Interpretation (Correlation)
        if r2 > 0.85: r2_desc = "Strong Correlation"
        elif r2 > 0.70: r2_desc = "Moderate Correlation"
        else: r2_desc = "Weak Correlation (Review Training Data)"

        # 2. RMSE Interpretation (The 10% Depth Rule)
        # Using 1.14m vs 15m as the 'High Precision' benchmark
        error_ratio = (rmse / max_depth) * 100
        if error_ratio < 5: rmse_desc = "Elite / Survey Grade"
        elif error_ratio < 10: rmse_desc = "High Precision"
        elif error_ratio < 20: rmse_desc = "Moderate / Acceptable"
        else: rmse_desc = "Low Precision (Check ACOLITE/Turbidity)"

        # 3. Bias Interpretation (Vertical Alignment)
        if abs(bias) < 0.10: bias_desc = "Near-Perfect Vertical Alignment"
        elif bias > 0.10: bias_desc = "Systemic Overestimation (Deep Bias)"
        else: bias_desc = "Systemic Underestimation (Shallow Bias)"

        return {
            "R2_Text": f"{r2:.3f} ({r2_desc})",
            "RMSE_Text": f"{rmse:.2f}m ({rmse_desc})",
            "Bias_Text": f"{bias:.2f}m ({bias_desc})"
        }
