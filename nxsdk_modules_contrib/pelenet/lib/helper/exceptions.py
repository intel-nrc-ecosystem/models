"""
@desc: Define custom error for parameter validity problems, used for network parameters
"""
class ParamaterNotValid(Exception):
   """Raised when a parameter is not valid"""
   pass

"""
@desc: Define custom error for argument validity problems, used for arguments of functions
"""
class ArgumentNotValid(Exception):
   """Raised when a argument is not valid"""
   pass
