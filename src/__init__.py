from qgis.core import QgsApplication
from .sdb_provider import SDBProvider

class SDBPlugin:
    def __init__(self, iface): # Added iface here
        self.iface = iface
        self.provider = None

    def initGui(self):
        # We create the provider instance here
        self.provider = SDBProvider()
        # Direct registration into the Processing Registry
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        # Clean up when the plugin is unchecked
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)

def classFactory(iface):
    return SDBPlugin(iface) # Pass iface to the class