#!/usr/bin/env python

import sys



results1 = {}
results2 = {}

## Process first file

data = open(sys.argv[1])

# Parse column names
headers = data.readline().strip().split(',')

# Process each entry
for line in data.readlines():
    columns = line.strip().split(',')
    data = dict(zip(headers, columns))

    block_size_x = int(data['Block Size X'])
    block_size_y = int(data['Block Size Y'])
    time_tile_size = int(data['Time Tile Size'])
    elems_per_thread = int(data['Elements/Thread'])
    gflops = float(data['Actual GFlop/s'])

    results1[(block_size_x, block_size_y, time_tile_size, elems_per_thread)] = gflops


## Process second file

data = open(sys.argv[2])

# Parse column names
headers = data.readline().strip().split(',')

# Process each entry
for line in data.readlines():
    columns = line.strip().split(',')
    data = dict(zip(headers, columns))

    block_size_x = int(data['Block Size X'])
    block_size_y = int(data['Block Size Y'])
    time_tile_size = int(data['Time Tile Size'])
    elems_per_thread = int(data['Elements/Thread'])
    gflops = float(data['Actual GFlop/s'])

    results2[(block_size_x, block_size_y, time_tile_size, elems_per_thread)] = gflops


max_1 = max(results1.values())
max_2 = max(results2.values())


print('BSX, BSY, TTS, Elems, GFlops (1), GFlops(2), Normalized (1), Normalized (2)')
for (bsx, bsy, tts, elems), first_gflops in results1.items():
    if (bsx, bsy, tts, elems) in results2:
        second_gflops = results2[(bsx, bsy, tts, elems)]
        norm_1 = first_gflops / max_1
        norm_2 = second_gflops / max_2
        print('%d,%d,%d,%d,%f,%f,%f,%f' % (bsx, bsy, tts, elems, first_gflops, second_gflops, norm_1, norm_2))
