from lib.helper.singleton import Singleton

"""
@desc: Class contains system values and functions
       This class is a singleton and can be read by any object
@note: In contrast to derived parameters, these values here are system variables
       which change their values while the system runs, derived parameters remain fixed
"""
@Singleton
class System():
    
    # Initialize the system singleton
    def __init__(self):
        # Defines some global runtime objects which are available for all modules
        self.datalog = None

    # Set data log object
    def setDatalog(self, datalog):
        self.datalog = datalog
