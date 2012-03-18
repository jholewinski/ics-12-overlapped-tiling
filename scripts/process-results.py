#!/usr/bin/env python

import math
import sys

try:
    import yaml
except:
    print('Please install PyYAML')
    sys.exit(1)


def query_by_index(data, index):
  if index >= len(data):
    return data[-1]
  else:
    return data[index]


if len(sys.argv) != 2:
  print('Usage: %s <arch-file>' % sys.argv[0])
  sys.exit(1)


# Define the program configuration type
class Configuration:
  def __init__(self):
    self.block_size_x = 1.0
    self.block_size_y = 1.0
    self.block_size_z = 1.0
    self.time_tile_size = 1.0
    self.elements_per_thread = 1.0
    self.expected_performance = 0.0
    self.actual_performance = 0.0

  def __str__(self):
    return ('Block Size: (%d, %d, %d)\tT: %d\tE: %d\tExpected: %f\tActual: %f'
      % (int(self.block_size_x), int(self.block_size_y),
        int(self.block_size_z), int(self.time_tile_size),
        int(self.elements_per_thread), self.expected_performance,
        self.actual_performance))

# Read the architecture description file
try:
  handle = open(sys.argv[1])
except IOError, e:
  print(e)
  sys.exit(1)

arch = yaml.load(handle)
handle.close()

# Parse architecture information
max_regs_per_block = float(arch['max_regs_per_block'])
shared_mem_per_sm = float(arch['shared_mem_per_sm'])
max_blocks_per_sm = float(arch['max_blocks_per_sm'])
max_warps_per_sm = float(arch['max_warps_per_sm'])
max_threads_per_sm = float(arch['max_threads_per_sm'])
warp_size = float(arch['warp_size'])
fp_throughput = arch['fp_throughput']
global_bandwidth = float(arch['global_bandwidth'])
shared_bandwidth = float(arch['shared_bandwidth'])
num_sm = float(arch['num_sm'])


# Set up configuration list
configurations = []

# Parse column names
headers = sys.stdin.readline().strip().split(',')

# Process each entry
for line in sys.stdin.readlines():
    columns = line.strip().split(',')
    data = dict(zip(headers, columns))

    # Parse the provided fields
    total_fp_per_block = float(data['Total FP/Block'])
    elems_per_thread = float(data['Elements/Thread'])
    global_loads_per_block = float(data['Global Loads/Block'])
    useful_fp_per_block = float(data['Useful FP/Block'])
    shared_size = float(data['Shared Size'])
    shared_stores_per_block = float(data['Shared Stores/Block'])
    global_stores_per_block = float(data['Global Stores/Block'])
    time_tile_size = float(data['Time Tile Size'])
    useful_ratio = float(data['Useful Ratio'])
    shared_loads_per_block = float(data['Shared Loads/Block'])
    actual_gflops = float(data['Actual GFlop/s'])
    num_regs = float(data['Register Usage'])

    block_size_x = float(data['Block Size X'])
    num_blocks_x = float(data['Num Blocks X'])
    if 'Block Size Y' not in data:
        block_size_y = 1.0
    else:
        block_size_y = float(data['Block Size Y'])

    if 'Num Blocks Y' not in data:
        num_blocks_y = 1.0
    else:
        num_blocks_y = float(data['Num Blocks Y'])

    if 'Block Size Z' not in data:
        block_size_z = 1.0
    else:
        block_size_z = float(data['Block Size Z'])

    if 'Num Blocks Z' not in data:
        num_blocks_z = 1.0
    else:
        num_blocks_z = float(data['Num Blocks Z'])


    # Derive some parameters based on the program and architecture parameters
    global_per_block = global_loads_per_block + global_stores_per_block
    shared_per_block = shared_loads_per_block + shared_stores_per_block
    regs_per_block = num_regs * block_size_x * block_size_y * block_size_z
    blocks_per_sm_from_regs = math.floor(max_regs_per_block / regs_per_block)
    blocks_per_sm_from_shared = math.floor(shared_mem_per_sm / shared_size)
    blocks_per_sm = min(blocks_per_sm_from_regs, blocks_per_sm_from_shared,
      max_blocks_per_sm)
    threads_per_block = block_size_x * block_size_y * block_size_z
    warps_per_block = threads_per_block / warp_size
    warps_per_sm = min(warps_per_block * blocks_per_sm, max_warps_per_sm)
    threads_per_sm = min(warps_per_sm * warp_size, max_threads_per_sm)


    ###=== Model the performance of this configuration ===###

    ### Compute Time
    # For our block distribution, what is our expected FP throughput
    exp_fp_throughput = query_by_index(fp_throughput, int(warps_per_sm)-1)*10

    # Determine needed giga-instructions
    total_ginstr = total_fp_per_block * 1e-9

    # Estimate total compute time
    compute_time = total_ginstr / exp_fp_throughput


    ### Global Memory Access Time
    # How much data do we need to pull to/from global memory?
    global_data = global_per_block * 4.0 * 1e-9

    # Assume our effective bandwidth is 1/num_sm of the total
    global_bandwidth_per_sm = global_bandwidth # / num_sm

    # Also, due to alignment constraints, we almost always need two
    # transactions
    global_bandwidth_per_sm = global_bandwidth_per_sm  / 2.0

    # Estimate global memory access time
    global_time = global_data / global_bandwidth_per_sm


    ### Shared Memory Access Time
    # How much shared data do we need to copy?
    shared_data = shared_per_block * 4.0 * 1e-9

    # Estimate shared memory access time
    shared_time = shared_data / shared_bandwidth



    ### Totals
    # Estimate total execution time as the time needed for the limiting
    # resource
    #exec_time = max(global_time, compute_time)
    exec_time = global_time + compute_time + shared_time

    # What is the "expected" device performance?
    exp_device_perf = total_fp_per_block / exec_time * 1e-9

    # Scale by useful ratio to get "expected" actual performance
    exp_actual_perf = exp_device_perf * useful_ratio


    # Build the Configuration instance
    conf = Configuration()
    conf.block_size_x = block_size_x
    conf.block_size_y = block_size_y
    conf.block_size_z = block_size_z
    conf.time_tile_size = time_tile_size
    conf.elements_per_thread = elems_per_thread
    conf.actual_performance = actual_gflops
    conf.expected_performance = exp_actual_perf

    configurations.append(conf)

    #if global_time > compute_time:
    #  print('Limiting factor: memory')
    #else:
    #  print('Limiting factor: compute')
    #print(conf)

# Determine max actual/expected performance
max_actual = 0.0
max_expected = 0.0
for c in configurations:
  max_actual = max(max_actual, c.actual_performance)
  max_expected = max(max_expected, c.expected_performance)

# Sort by actual performance
confs_by_actual_perf = sorted(configurations,
  key=lambda conf: conf.actual_performance)
confs_by_actual_perf.reverse()
print('Sorted by Actual Performance (Max Expected: %f):' % max_expected)
for c in confs_by_actual_perf:
  print(c)

print('\n')

# Sort by expected performance
confs_by_expected_perf = sorted(configurations,
  key=lambda conf: conf.expected_performance)
confs_by_expected_perf.reverse()
print('Sorted by Expected Performance (Max Actual: %f):' % max_actual)
for c in confs_by_expected_perf:
  print(c)



