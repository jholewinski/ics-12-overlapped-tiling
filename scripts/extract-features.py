#!/usr/bin/env python

import optparse
import sys

parser = optparse.OptionParser()
parser.add_option('-x', '--block-size-x', dest='blockSizeX',
                  action='store', type='int',
                  help='Filter by block size (X)')
parser.add_option('-y', '--block-size-y', dest='blockSizeY',
                  action='store', type='int',
                  help='Filter by block size (Y)')
parser.add_option('-p', '--header', dest='header',
                  action='store_true', default=False,
                  help='Print column headers')


(options, args) = parser.parse_args()

# Parse column names
headers = sys.stdin.readline().strip().split(',')

# Print output headers
if options.header:
#    sys.stdout.write('total_fp,elem_per_thread,gld_per_block,useful_fp,')
#    sys.stdout.write('shared_size,shared_st_per_block,gst_per_block,')
#    sys.stdout.write('time_tile_size,block_size_x,block_size_y,block_size_z,useful_ratio,')
#    sys.stdout.write('shared_ld_per_block,')
#    sys.stdout.write('useful_fp_per_gldgst,')
#    sys.stdout.write('useful_fp_per_shared,')
#    sys.stdout.write('ratio1,')
#    sys.stdout.write('total_fp_per_gldgst,')
#    sys.stdout.write('total_fp_per_shared,')
#    sys.stdout.write('ratio2,')
#    sys.stdout.write('useful_fp_per_thread_per_gldgst,')
#    sys.stdout.write('gld_per_thread, gst_per_thread,')
#    sys.stdout.write('glb_trans_per_block,')
#    sys.stdout.write('shared_over_global,')
#    sys.stdout.write('glb_trans_times_useful_ratio,')
    sys.stdout.write('block_size_x,')
    sys.stdout.write('block_size_y,')
    sys.stdout.write('time_tile_size,')
    sys.stdout.write('elements_per_thread,')
    sys.stdout.write('global_ld_block,')
    sys.stdout.write('global_st_block,')
    sys.stdout.write('shared_ld_block,')
    sys.stdout.write('shared_st_block,')
    sys.stdout.write('ops_per_load,')
    sys.stdout.write('dimensionality,')
    sys.stdout.write('shared_over_global,')
    sys.stdout.write('num_blocks_x,')
    sys.stdout.write('num_blocks_y,')
    sys.stdout.write('num_blocks_z,')
    sys.stdout.write('glb_trans_per_block,')
    sys.stdout.write('num_arrays,')
    sys.stdout.write('actual_gflops\n')

for line in sys.stdin.readlines():
    columns = line.strip().split(',')
    data = dict(zip(headers, columns))
    
    total_fp = int(data['Total FP'])
    elem_per_thread = int(data['Elements/Thread'])
    gld_per_block = int(data['Global Loads/Block'])
    useful_fp = int(data['Useful FP'])
    shared_size = int(data['Shared Size'])
    shared_st_per_block = int(data['Shared Stores/Block'])
    gst_per_block = int(data['Global Stores/Block'])
    time_tile_size = int(data['Time Tile Size'])
    block_size_x = int(data['Block Size X'])
    useful_ratio = float(data['Useful Ratio'])
    shared_ld_per_block = int(data['Shared Loads/Block'])
    num_arrays = int(data['num_arrays'])
    actual_gflops = float(data['Actual GFlop/s'])
    ops_per_load = float(data['ops_per_load'])
    dimensionality = float(data['dimensionality'])
    num_blocks_x = int(data['Num Blocks X'])

    if 'Block Size Y' not in data:
        block_size_y = 1
    else:
        block_size_y = int(data['Block Size Y'])

    if 'Num Blocks Y' not in data:
        num_blocks_y = 1
    else:
        num_blocks_y = int(data['Num Blocks Y'])

    if 'Block Size Z' not in data:
        block_size_z = 1
    else:
        block_size_z = int(data['Block Size Z'])

    if 'Num Blocks Z' not in data:
        num_blocks_z = 1
    else:
        num_blocks_z = int(data['Num Blocks Z'])

    if options.blockSizeX and 'Block Size Y' in data and options.blockSizeX != block_size_x:
        continue

    if options.blockSizeY and 'Block Size Z' in data and options.blockSizeY != block_size_y:
        continue

    useful_fp_per_gldgst = float(useful_fp) / float(gld_per_block+gst_per_block)
    useful_fp_per_shared = float(useful_fp) / float(max(shared_ld_per_block+shared_st_per_block,1))
    ratio1 = useful_fp_per_gldgst / useful_fp_per_shared

    total_fp_per_gldgst = float(total_fp) / float(gld_per_block+gst_per_block)
    total_fp_per_shared = float(total_fp) / float(max(shared_ld_per_block+shared_st_per_block,1))
    ratio2 = total_fp_per_gldgst / total_fp_per_shared

    useful_fp_per_thread_per_gldgst = float(useful_fp) / float(block_size_y*block_size_y) / float(gld_per_block+gst_per_block)

    gld_per_thread = float(gld_per_block) / float(block_size_x*block_size_y)
    gst_per_thread = float(gst_per_block) / float(block_size_x*block_size_y)

    glb_trans_per_block = float(block_size_x) / 16.0 * float(block_size_y) * (gld_per_thread + gst_per_thread)

    shared_over_global = float(shared_ld_per_block+shared_st_per_block) / float(gld_per_block+gst_per_block)

    glb_trans_times_useful_ratio = glb_trans_per_block * useful_ratio

    global_per_block = gld_per_block + gst_per_block
    shared_per_block = shared_ld_per_block + shared_st_per_block

    sys.stdout.write(str(block_size_x))
    sys.stdout.write(',')
    sys.stdout.write(str(block_size_y))
    sys.stdout.write(',')
    sys.stdout.write(str(time_tile_size))
    sys.stdout.write(',')
    sys.stdout.write(str(elem_per_thread))
    sys.stdout.write(',')
    sys.stdout.write(str(gld_per_block))
    sys.stdout.write(',')
    sys.stdout.write(str(gst_per_block))
    sys.stdout.write(',')
    sys.stdout.write(str(shared_ld_per_block))
    sys.stdout.write(',')
    sys.stdout.write(str(shared_st_per_block))
    sys.stdout.write(',')
    sys.stdout.write(str(ops_per_load))
    sys.stdout.write(',')
    sys.stdout.write(str(dimensionality))
    sys.stdout.write(',')
    sys.stdout.write(str(shared_over_global))
    sys.stdout.write(',')
    sys.stdout.write(str(num_blocks_x))
    sys.stdout.write(',')
    sys.stdout.write(str(num_blocks_y))
    sys.stdout.write(',')
    sys.stdout.write(str(num_blocks_z))
    sys.stdout.write(',')
    sys.stdout.write(str(glb_trans_per_block))
    sys.stdout.write(',')

 
#    sys.stdout.write(str(total_fp))
#    sys.stdout.write(',')
#    sys.stdout.write(str(gld_per_block))
#    sys.stdout.write(',')
#    sys.stdout.write(str(useful_fp))
#    sys.stdout.write(',')
#    sys.stdout.write(str(shared_size))
#    sys.stdout.write(',')
#    sys.stdout.write(str(shared_st_per_block))
#    sys.stdout.write(',')
#    sys.stdout.write(str(gst_per_block))
#    sys.stdout.write(',')
#    sys.stdout.write(str(block_size_z))
#    sys.stdout.write(',')
#    sys.stdout.write(str(useful_ratio))
#    sys.stdout.write(',')
#    sys.stdout.write(str(shared_ld_per_block))
#    sys.stdout.write(',')
#    sys.stdout.write(str(useful_fp_per_gldgst))
#    sys.stdout.write(',')
#    sys.stdout.write(str(useful_fp_per_shared))
#    sys.stdout.write(',')
#    sys.stdout.write(str(ratio1))
#    sys.stdout.write(',')
#    sys.stdout.write(str(total_fp_per_gldgst))
#    sys.stdout.write(',')
#    sys.stdout.write(str(total_fp_per_shared))
#    sys.stdout.write(',')
#    sys.stdout.write(str(ratio2))
#    sys.stdout.write(',')
#    sys.stdout.write(str(useful_fp_per_thread_per_gldgst))
#    sys.stdout.write(',')
#    sys.stdout.write(str(gld_per_thread))
#    sys.stdout.write(',')
#    sys.stdout.write(str(gst_per_thread))
#    sys.stdout.write(',')
#    sys.stdout.write(str(glb_trans_per_block))
#    sys.stdout.write(',')
#    sys.stdout.write(str(shared_over_global))
#    sys.stdout.write(',')
#    sys.stdout.write(str(glb_trans_times_useful_ratio))
#    sys.stdout.write(',')
    sys.stdout.write(str(num_arrays))
    sys.stdout.write(',')
    sys.stdout.write(str(actual_gflops))
    sys.stdout.write('\n')
    
