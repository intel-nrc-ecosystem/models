from datetime import datetime
from re import match
import logging
import os

from nxsdk.logutils.nxlogging import set_verbosity

"""
@desc: Class to prepare logging and data storge
       logging: system and loihi
       storage: data, plots and results
"""
class Datalog():

    """
    @desc: Initialize system class with system parameters and function calls
    """
    def __init__(self, parameters, name=''):
        # Define important system values
        self.p = parameters
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dirname = '' if name == '' else '_'+self.filenameSave(name) 
        self.dir = self.p.dataLogPath + time + dirname + '/'

        # Initialize datalog
        self.initDataLog()

        # Initialize logging
        self.initLogging()

        # Store parameters in file
        self.writeParameters()

        # Log successful configuration of the system
        logging.info('System initialized')

    """
    @desc: Convert arbitrary string to filename save string
    """
    def filenameSave(self, str):
        # Only allow letters ("a-z" & "A-Z"), numbers "0-9" and "-", "_"
        return "".join([c for c in str if match(r'[a-zA-Z0-9-_]', c)])

    """
    @desc: Init datalog
    """
    def initDataLog(self):
        # Create folders in datalog directory
        os.mkdir(self.dir)
        os.mkdir(self.dir + 'data')
        os.mkdir(self.dir + 'plots')
        os.mkdir(self.dir + 'results')

    """
    @desc: Initialize and configure logging system
    """
    def initLogging(self):
        # Define filename and filepath
        filepath = self.dir + '/system.log'

        # Config logging
        logging.basicConfig(
            filename = filepath,
            level = self.p.systemLoggingLevel,
            format = '%(asctime)s %(levelname)s: %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S'
        )

        # Set Loihi logging level
        set_verbosity(self.p.loihiLoggingLevel)

    """
    @desc: Write parameters to file
    """
    def writeParameters(self):
        # Define filename and filepath
        filepath = self.dir + '/parameters.log'

        # Open file for writing, overwrite content if exists
        f = open(filepath, "w")
        # Write parameter table to file
        f.write(self.p.__str__())
        # Close file buffer
        f.close()
