#!/usr/bin/env python

import sys

runs = {}

for line in sys.stdin.readlines():
    line = line.strip()
    (run, data) = line.split('#')
    run = int(run)
    data = data.strip()
    (key, value) = data.split(':')
    key = key.strip()
    value = value.strip()
    
    if run not in runs:
        runs[run] = {}

    runs[run][key] = value

    


sys.stdout.write('Arithmetic Intensity,Global Accesses,Shared Accesses,Useful FP Ratio,Global Trans Per Point,GFlop/s\n')

for (run, values) in runs.iteritems():
    blockSizeX = int(values['Block Size X'])
    blockSizeY = int(values['Block Size Y'])
    numBlocksX = int(values['Num Blocks X'])
    numBlocksY = int(values['Num Blocks Y'])
    totalBlocks = numBlocksX * numBlocksY
    gldPerBlock = int(values['Global Loads/Block'])
    gstPerBlock = int(values['Global Stores/Block'])
    sldPerBlock = int(values['Shared Loads/Block'])
    sstPerBlock = int(values['Shared Stores/Block'])
    
    globalMemAccesses = (gldPerBlock+gstPerBlock)*totalBlocks
    sharedMemAccesses = (sldPerBlock+sstPerBlock)*totalBlocks

    globalTransPerPoint = (blockSizeX / 16) * blockSizeY

    usefulFPRatio = float(values['Useful Ratio'])

    sys.stdout.write('%s,%d,%d,%f,%d,%s\n' % 
                     (values['Arithmetic Intensity'],
                      globalMemAccesses,
                      sharedMemAccesses,
                      usefulFPRatio,
                      globalTransPerPoint,
                      values['Actual GFlop/s']))

sys.stdout.flush()

    
            


