# -*- coding: utf-8 -*-
import os
from qgis.core import QgsProcessingProvider
# In QGIS 4/Qt6, ensure we import QIcon from PyQt6.QtGui
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
        # Explicitly point to your icon.png in the plugin folder
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        # Fallback to the default if the file is missing
        return super().icon()
    
