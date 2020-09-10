import os
import logging
from nxsdk.logutils.nxlogging import LoggingLevel

"""
@desc: Include parameters regarding the system (e.g. error/warning thresholds)
"""
def includeSystem(self):

    # Plot styles
    self.pltColor = '#000000'
    self.pltLegendSize = 12
    self.pltFontFamily = 'CMU Sans Serif'
    self.pltFontSize = 14
    self.pltFileType = 'svg'  # png, svg, jpg, pdf
    self.pltAxesGrid = True
    self.pltLegendFancybox = True
    self.pltLegendFramealpha = 0.75
    self.pltPatchLinewidth = 0

    # Some descrete plot colors
    self.pltColor1 = '#b74d41'
    self.pltColor2 = '#41aab7'
    self.pltColor3 = '#41b74d'
    self.pltColor4 = '#b78841'

    # Data log
    self.loihiLoggingLevel = LoggingLevel.INFO
    self.systemLoggingLevel = logging.INFO
    self.dataLogPath = os.path.join(os.getcwd(), 'log/')  # Define datalog path
    self.snipsPath = os.path.join(os.getcwd(), 'pelenet/snips/')  # Path to Loihi snip C files
    self.targetPath = os.path.join(os.getcwd(), 'data/robot/')  # Path to movement trajectory
    self.expLogPath = None  # The explicit path within dataLogPath, which is set when parameter instance is created, it depends on datetime

    # Loihi chip
    self.numChips = 2  # Number of cores available in current setup
    self.numCoresPerChip = 128  # Number of cores per chip in current setup
    self.neuronsPerCore = 20  # Numer of neurons distributed to each Loihi core
    self.bufferFactor = 20  # Number of timesteps the buffer should collect spike data from the Loihi cores
    self.numTagBits = 0  # Bits available for tag precision
    self.numDelayBits = 0  # Bits available for synaptic delay precision
    self.numWeightBits = 8  # Bits available for weight precision

    # Validity check
    self.maxSparseToDenseLimit = 6000  # When e.g. doing an imshow plot a normal dense matrix is necessary, which only can be transformed from a sparse matrix, when memory is sufficient
