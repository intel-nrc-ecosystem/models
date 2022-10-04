#INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
#Copyright © 2021 Intel Corporation.
#
#This software and the related documents are Intel copyrighted
#materials, and your use of them is governed by the express
#license under which they were provided to you (License). Unless
#the License provides otherwise, you may not use, modify, copy,
#publish, distribute, disclose or transmit  this software or the
#related documents without Intel's prior written permission.
#
#This software and the related documents are provided as is, with
#no express or implied warranties, other than those that are
#expressly stated in the License.

import sys, os
import glob
import numpy as np
import h5py
import inspect

class NTIDIGITSDataset():
    def __init__(self, path='data', train=True):
        datasetPath = path + '/n-tidigits.hdf5'

        print(inspect.cleandoc('''
            Neuromorphic TIDIGITS dataset is the work of Sensors Group, Institute of 
            Neuromoroinformatics, Zurich. It is publicly available under Creative 
            Commons Attribution-ShareAlike 4.0 International License here: 
            http://sensors.ini.uzh.ch/databases.html
            
            Please cite the following paper if you make use of it.\n
            Jithendar Anumula, Daniel Neil, Tobi Delbruck and Shih-Chii Liu, 
            "Feature Representations for Neuromorphic Audio Spike Streams." 
            Front. Neurosci., vol. 12, p. 23, 2018.\n\n
        '''))

        # check if dataset is available in path. If not download it
        if len(glob.glob(datasetPath)) == 0: # dataset does not exist
            os.makedirs(path, exist_ok=True)
            
            print('Dataset not available locally. Starting download ...')
            os.system('wget -P ' + path +'/ https://www.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5')
            print('Completed download.')

        self.data = h5py.File(datasetPath, 'r')
        self.prefixStr = 'train' if train==True else 'test'
        self.label = {}
        for i in range(11):
            if i==0:    self.label['z'] = 0
            elif i==10: self.label['o'] = 10
            else:       self.label[str(i)] = i

        self.samples = []
        for labelStr in self.data[self.prefixStr + '_labels']:
            labelStr = labelStr.decode('utf-8')
            seq = labelStr[labelStr.rfind('-')+1:]

            if len(seq) == 1: # consider only single utterance samples
                self.samples.append([labelStr, self.label[seq]])

    def __getitem__(self, index):
        key, label = self.samples[index]
        
        addr      = np.array(self.data[self.prefixStr + '_addresses'] [key])
        timeStamp = np.array(self.data[self.prefixStr + '_timestamps'][key]).astype(float) * 1000 # convert to ms

        event = np.zeros((len(addr), 4))
        event[:, 0] = addr
        event[:, 3] = timeStamp

        return event, label
    
    def __len__(self):
        return len(self.samples)


        
