from nxsdk.graph.processes.phase_enums import Phase
import logging

"""
@desc: Add SNIPs to the chips of the system

TODO: Only add SNIPs to used chips, not to all available chips on the system
"""
def addResetSnips(self, board):
    # Add one SNIPs to every chip
    resetInitSnips = []
    for i in range(self.p.numChips):
        # SNIP for initializing some values for the reset SNIP
        resetInitSnips.append(board.createSnip(
            name='init'+str(i),
            cFilePath=self.p.snipsPath + "/reset_init.c",
            includeDir=self.p.snipsPath,
            funcName='initialize_reset',
            phase=Phase.EMBEDDED_INIT,
            lmtId=0,
            chipId=i))

    # SNIPs for resetting the voltages and currents
    resetSnips = []
    for i in range(self.p.numChips):
        # Add one SNIP for every chip
        board.createSnip(
            name='reset'+str(i),
            cFilePath=self.p.snipsPath + "/reset.c",
            includeDir=self.p.snipsPath,
            guardName='do_reset',
            funcName='reset',
            phase=Phase.EMBEDDED_MGMT,
            lmtId=0,
            chipId=i)

    logging.info('SNIPs added to chips')

    return resetInitSnips

"""
@desc: Create and connect channels for initializing values for the reset SNIPs
"""
def createAndConnectResetInitChannels(self, board, resetInitSnips):
    resetInitChannels = []
    # Add one channel to every chip
    for i in range(self.p.numChips):
        # Create channel for init data with buffer size of 3
        initResetChannel = board.createChannel(bytes('initreset'+str(i), 'utf-8'), "int", 3)
        
        # Connect channel to init snip
        initResetChannel.connect(None, resetInitSnips[i])

        # Add channel to list
        resetInitChannels.append(initResetChannel)
    
    logging.info('Channels added')

    return resetInitChannels
