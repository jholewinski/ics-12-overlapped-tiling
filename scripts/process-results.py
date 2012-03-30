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
fp_throughput_all_sm = arch['fp_throughput_all_sm']
global_bandwidth = float(arch['global_bandwidth'])
shared_bandwidth = float(arch['shared_bandwidth'])*16.0
num_sm = float(arch['num_sm'])


# Set up excel dump
excel_dump = open('dump.csv', 'w')

dump_columns = [
  'block_size_x',
  'block_size_y',
  'block_size_z',
  'elems_per_thread',
  'time_tile_size',
  'elapsed_time',
  'actual_gflops',
  'device_gflops',
  'num_regs',
  'shared_loads_per_block',
  'shared_stores_per_block',
  'global_loads_per_block',
  'global_stores_per_block',
  'useful_fp_per_block',
  'total_fp_per_block',
  'useful_ratio',
  'shared_size',
  'warps_per_sm',
  'threads_per_sm',
  'total_blocks',
  'shared_loads_per_step',
  'shared_stores_per_step',
  'total_fp_per_step',
  'adj_bandwidth',
  'global_data',
  'global_time',
  'shared_time',
  'exp_fp_throughput',
  'compute_time',
  'limiting_time_step_1',
  'global_time',
  'shared_load_time',
  'shared_store_time',
  'shared_time',
  'exp_fp_throughput',
  'compute_time',
  'limiting_time_step_n',
  'adj_bandwidth',
  'global_data',
  'global_time',
  'limiting_time_final_step',
  'exec_time',
  'exp_device_perf',
  'exp_actual_perf'
]

excel_dump.write('%s,\n' % ','.join(dump_columns))

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
    device_gflops = float(data['Device GFlop/s'])
    num_regs = float(data['Register Usage'])
    elapsed_time = float(data['Elapsed Time'])

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


    excel_dump.write('%d,' % int(block_size_x))
    excel_dump.write('%d,' % int(block_size_y))
    excel_dump.write('%d,' % int(block_size_z))
    excel_dump.write('%d,' % int(elems_per_thread))
    excel_dump.write('%d,' % int(time_tile_size))
    excel_dump.write('%f,' % elapsed_time)
    excel_dump.write('%f,' % actual_gflops)
    excel_dump.write('%f,' % device_gflops)
    excel_dump.write('%d,' % int(num_regs))
    excel_dump.write('%d,' % int(shared_loads_per_block))
    excel_dump.write('%d,' % int(shared_stores_per_block))
    excel_dump.write('%d,' % int(global_loads_per_block))
    excel_dump.write('%d,' % int(global_stores_per_block))
    excel_dump.write('%d,' % int(useful_fp_per_block))
    excel_dump.write('%d,' % int(total_fp_per_block))
    excel_dump.write('%f,' % useful_ratio)
    excel_dump.write('%d,' % int(shared_size))

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
    total_blocks = num_blocks_x * num_blocks_y * num_blocks_z

    excel_dump.write('%d,' % int(warps_per_sm))
    excel_dump.write('%d,' % int(threads_per_sm))
    excel_dump.write('%d,' % int(total_blocks))


    ###=== Model the performance of this configuration ===###

    if time_tile_size > 1:
      shared_loads_per_step = total_blocks * shared_loads_per_block / (time_tile_size-1)
    else:
      shared_loads_per_step = 0.0
    shared_stores_per_step = total_blocks * shared_stores_per_block / time_tile_size
    total_fp_per_step = total_blocks * total_fp_per_block / time_tile_size

    excel_dump.write('%d,' % int(shared_loads_per_step))
    excel_dump.write('%d,' % int(shared_stores_per_step))
    excel_dump.write('%d,' % int(total_fp_per_step))


    memory_scale = warps_per_sm / max_warps_per_sm
    adj_global_bandwidth = global_bandwidth * memory_scale

    ### Step 1

    # For initial loads, we assume 50% bandwidth
    adj_bandwidth = adj_global_bandwidth #/ 2.0
    excel_dump.write('%f,' % adj_bandwidth)
    # How much data do we need for the kernel? (GBytes)
    global_data = total_blocks * global_loads_per_block * 4.0 * 1e-9
    excel_dump.write('%f,' % global_data)
    # Estimate global memory access time
    global_time = global_data / adj_bandwidth
    excel_dump.write('%f,' % global_time)

    # How long will it take to complete the stores
    shared_time = shared_stores_per_step * 4.0 * 1e-9 / shared_bandwidth
    excel_dump.write('%f,' % shared_time)

    # What is our expected FP throughput?
    exp_fp_throughput = query_by_index(fp_throughput_all_sm,
      int(warps_per_sm)-1)
    excel_dump.write('%f,' % exp_fp_throughput)
    # How long will compute take?
    compute_time = total_fp_per_step * 1e-9 / exp_fp_throughput
    excel_dump.write('%f,' % compute_time)

    # What is the limiting resource?
    limiting_time_step_1 = max(global_time, shared_time, compute_time)
    excel_dump.write('%f,' % limiting_time_step_1)

    #print('Step1: %f, %f, %f, %f' % (
    #  global_time, shared_time, compute_time, limiting_time_step_1))

    ### Step n > 1

    # We do not access global memory here
    global_time = 0.0
    excel_dump.write('%f,' % global_time)

    # What is the shared memory time?
    shared_load_time = shared_loads_per_step * 4.0 * 1e-9 / shared_bandwidth
    excel_dump.write('%f,' % shared_load_time)
    shared_store_time = shared_stores_per_step * 4.0 * 1e-9 / shared_bandwidth
    excel_dump.write('%f,' % shared_store_time)
    shared_time = shared_load_time + shared_store_time
    excel_dump.write('%f,' % shared_time)

    # What is our expected FP throughput?
    exp_fp_throughput = query_by_index(fp_throughput_all_sm,
      int(warps_per_sm)-1)
    excel_dump.write('%f,' % exp_fp_throughput)
    # How long will compute take?
    compute_time = total_fp_per_step * 1e-9 / exp_fp_throughput
    excel_dump.write('%f,' % compute_time)

    # What is the limiting resource?
    limiting_time_step_n = max(global_time, shared_time, compute_time)
    excel_dump.write('%f,' % limiting_time_step_n)


    ### Final write

    # Here, we only write to global memory
    # For initial loads, we assume 60% bandwidth
    adj_bandwidth = adj_global_bandwidth * 0.60
    excel_dump.write('%f,' % adj_bandwidth)
    # How much data do we need for the kernel? (GBytes)
    global_data = total_blocks * global_stores_per_block * 4.0 * 1e-9
    excel_dump.write('%f,' % global_data)
    # Estimate global memory access time
    global_time = global_data / adj_bandwidth
    excel_dump.write('%f,' % global_time)

    limiting_time_final_step = global_time
    excel_dump.write('%f,' % limiting_time_final_step)


    ### Total expected time
    exec_time = limiting_time_step_1 + (time_tile_size-1)*limiting_time_step_n + limiting_time_final_step
    excel_dump.write('%f,' % exec_time)

    # What is the "expected" device performance?
    exp_device_perf = total_fp_per_block * total_blocks / exec_time * 1e-9
    excel_dump.write('%f,' % exp_device_perf)

    # Scale by useful ratio to get "expected" actual performance
    exp_actual_perf = exp_device_perf * useful_ratio
    excel_dump.write('%f,' % exp_actual_perf)


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

    excel_dump.write('\n')

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


excel_dump.close()

