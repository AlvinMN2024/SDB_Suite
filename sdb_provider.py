# -*- coding: utf-8 -*-
"""Register the SDB Production Suite algorithms with QGIS Processing."""
import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .sdb_trainer_alg import SDBTrainerAlgorithm
from .sdb_generator_alg import SDBGeneratorAlgorithm


class SDBProvider(QgsProcessingProvider):
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
