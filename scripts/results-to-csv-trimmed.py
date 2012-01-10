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

    


sys.stdout.write('Useful FP Per Block,FP Per Global Access,Global Mem Accesses,Shared Mem Accesses,Actual GFlop/s\n')

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
    
    usefulFPPerBlock = int(values['Useful FP'])
    totalFPPerBlock = int(values['Total FP'])

    globalMemAccesses = (gldPerBlock+gstPerBlock) # *totalBlocks
    sharedMemAccesses = (sldPerBlock+sstPerBlock) # *totalBlocks

    globalTransPerPoint = (blockSizeX / 16) * blockSizeY

    fpPerGlobalAccess = float(usefulFPPerBlock) / float(gldPerBlock+gstPerBlock)
    allFPPerGlobalAccess = float(totalFPPerBlock) / float(gldPerBlock+gstPerBlock)

    usefulFPRatio = float(values['Useful Ratio'])

    sys.stdout.write('%d,%f,%d,%d,%s\n' % 
                     (usefulFPPerBlock,
                      fpPerGlobalAccess,
                      globalMemAccesses,
                      sharedMemAccesses,
                      values['Actual GFlop/s']))

sys.stdout.flush()

    
            


