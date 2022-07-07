import sys
import os
import numpy as np
import nxsdk.api.n2a as nx
import scipy.sparse as sps
from nxsdk_modules.slayer.src.slayer2loihi import Slayer2Loihi as s2l
from nxsdk_modules.dvs.src.dvs import DVS
import curses
import inspect
import errno
import subprocess

"""The live DVS Gesture demo"""

def printResult(names, result, stdscr):
    """
    Simple text based printout of results
    """
    count = np.sum(result, axis=0)
    max_count = np.max(count)
    
    if max_count>40:
        count = count*40//max_count
    
    for (ii, name) in enumerate(names):
        lineString = ' '
        for jj in range(count[ii]):
            lineString = lineString + '-'
        lineString = lineString + '>'
        stdscr.addstr(ii, 0, name + "\t" + lineString + "\t\t\t\t\t\t\t\t")
    
    stdscr.refresh()



def startVisualizer():
    """
    Helper function to launch the DVS visualizer for 128x128 pixels only
    """
    path = os.path.dirname(inspect.getfile(startVisualizer))

    # compile the visualizer
    subprocess.run(["gcc "
                    "-O3 "
                    + "$(sdl2-config --cflags) "
                    + path + "/visualizer/visualizer_DVS128.c "
                    + "$(sdl2-config --libs) "
                    + "-o "
                    + path + "/visualize_DVS128_spikes"],  shell=True)

    # setup spike fifo
    spikeFifoPath = path + "/spikefifo"

    try:
        os.mkfifo(spikeFifoPath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

    # this environment variables sets where Loihi spikes will appear on the host
    os.environ['NX_SPIKE_OUTPUT'] = spikeFifoPath

    # run the visualizer
    subprocess.Popen(
        [path + "/visualize_DVS128_spikes", "--source=" + spikeFifoPath])

loadState = False
saveState = False
boardName = 'dvs_gesture'

if 'PARTITION' in os.environ:
    del os.environ['PARTITION']
if 'SLURM' in os.environ:
    del os.environ['SLURM']
os.environ['KAPOHOBAY'] = "1"

#cheating because I know the number of classes here
classes = [None]*11
classes[0] = 'clapping'
classes[1] = 'right wave'
classes[2] = 'left wave'
classes[3] = 'right arm CW'
classes[4] = 'right arm CCW'
classes[5] = 'left arm CW'
classes[6] = 'left arm CCW'
classes[7] = 'arm roll'
classes[8] = 'drums\t'
classes[9] = 'air guitar'
classes[10] = 'other\t'


startVisualizer()

modelPath = s2l.getModels() + '/03_IBMGesture/'

# The NeuroCore from which we'll start placing compartments
corenum = 0 

compProto = s2l.compartmentPrototype(modelPath+'network.yaml')

# create an empty network
net = nx.NxNet()

inputSpec = dict()
inputSpec["sizeX"] = 128
inputSpec["sizeY"] = 128
inputSpec["sizeC"] = 2
compartmentsPerCore = 1024


# instantiate a DVS spike gen. Actual wiring will be taken care of by a snip
dvs = DVS(net=net,
          dimX=240,
          dimY=180,
          dimP=2)

layerInput, inputConnectionGroup, corenum = s2l.inputLayer(net, inputSpec, corenum, compartmentsPerCore)

# Make some spike output ports for visualization
opg = net.createSpikeOutputPortGroup(size=128*128*2)

layerInput.connect(opg, connectionMask=sps.identity(128*128*2))

poolSpec = dict()
poolSpec["stride"] = 4
poolSpec["compProto"] = compProto
poolSpec["weightFile"] = modelPath + 'Trained/pool1.npy'
compartmentsPerCore = 4096/16

layer1, corenum = s2l.poolingLayer(layerInput, poolSpec, corenum, compartmentsPerCore)


convSpec = dict()
convSpec["compProto"] = compProto
convSpec["dimX"] = 5
convSpec["dimY"] = 5
convSpec["dimC"] = 16
convSpec["weightFile"] = modelPath + 'Trained/conv1.npy'
compartmentsPerCore = 4096/4

layer2, corenum = s2l.convLayer(layer1, convSpec, corenum, compartmentsPerCore)


poolSpec = dict()
poolSpec["stride"] = 2
poolSpec["compProto"] = compProto
poolSpec["weightFile"] = modelPath + 'Trained/pool2.npy'
compartmentsPerCore = 256 

layer3, corenum = s2l.poolingLayer(layer2, poolSpec, corenum, compartmentsPerCore)

convSpec = dict()
convSpec["compProto"] = compProto
convSpec["dimX"] = 3
convSpec["dimY"] = 3
convSpec["dimC"] = 32
convSpec["weightFile"] = modelPath + 'Trained/conv2.npy'
compartmentsPerCore = 130

layer4, corenum = s2l.convLayer(layer3, convSpec, corenum, compartmentsPerCore)


poolSpec = dict()
poolSpec["stride"] = 2
poolSpec["compProto"] = compProto
poolSpec["weightFile"] = modelPath + 'Trained/pool3.npy'
compartmentsPerCore = 256

layer5, corenum = s2l.poolingLayer(layer4, poolSpec, corenum, compartmentsPerCore)

#re-order the compartment group to SlayerPyTorch convention before the fully connected layers
layer5 = s2l.reorderLayer(layer5)


fullSpec = dict()
fullSpec["compProto"] = compProto
fullSpec["dim"] = 512
fullSpec["weightFile"] = modelPath + 'Trained/fc1.npy'
compartmentsPerCore = 60

layer6, corenum  = s2l.fullLayer(layer5, fullSpec, corenum, compartmentsPerCore)


fullSpec = dict()
fullSpec["compProto"] = compProto
fullSpec["dim"] = 11
fullSpec["weightFile"] = modelPath + 'Trained/fc2.npy'
compartmentsPerCore = 20

layerOutput, corenum = s2l.fullLayer(layer6, fullSpec, corenum, compartmentsPerCore)

dummyProbes = s2l.setupSpikeCounters(layerOutput)

if loadState is False:
    print("Compiling")
    compiler = nx.N2Compiler()
    board = compiler.compile(net)
else:
    board = s2l.initBoard(boardName)

sampleLength = 200

s2l.writeHeader(inputConnectionGroup, layerOutput, 0, sampleLength)

spikeCntrChannel = s2l.prepSpikeCounter(board, 100, int(corenum))

#Custom injection snip here
snip = os.path.abspath(os.path.dirname(inspect.getfile(printResult))) + '/snips/gesture.c'
funcName = "dvs_snip_injection"
guardName = "do_dvs_snip_injection"

dvs.setupSnips(board, 
               snip=snip, 
               funcName=funcName, 
               guardName=guardName)

board.start()
if saveState is True:
    print("Saving Board")
    s2l.saveBoard(board, boardName)
if loadState is True:
    print("Loading Board")
    s2l.loadBoard(board, boardName)

numSteps = 100000000
board.run(numSteps, aSync=True)

stdscr = curses.initscr()
curses.noecho()
curses.cbreak()

result = np.zeros( (10,layerOutput.numNodes), dtype=int)

jj = 0
for ii in range(int(numSteps/sampleLength)):
    result[jj] = np.array(s2l.getResults(spikeCntrChannel, 1, layerOutput.numNodes, dummyProbes, saveResults=False)[0])
    jj = jj + 1
    if jj == 10:
        jj=0
    printResult(classes, result, stdscr)

board.finishRun()
board.disconnect()
