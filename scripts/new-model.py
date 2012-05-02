#!/usr/bin/env python

import sys
import yaml

# Global data
class GlobalData:
  def __init__(self, config):

    self.trapezoid_per_stage = float(config['trapezoid_per_stage'])
    self.concurrent_blocks = float(config['concurrent_blocks'])
    self.mem_latency = float(config['mem_latency'])
    self.mem_pipeline_depth = float(config['mem_pipeline_depth'])
    self.bandwidth = float(config['bandwidth'])
    self.clock = float(config['clock'])
    self.coalesce_degree = float(config['coalesce_degree'])
    # Bytes per data element
    self.data_size = float(config['data_size'])

    self.real_per_block_x = float(config['real_per_block_x'])
    self.real_per_block_y = float(config['real_per_block_y'])
    self.elems_per_block_x = float(config['elems_per_block_x'])
    self.elems_per_block_y = float(config['elems_per_block_y'])

    self.time_tile_size = float(config['time_tile_size'])
    self.elems_per_op = float(config['elems_per_op'])
    self.num_arrays = float(config['num_arrays'])

    self.num_warp_instr = float(config['num_warp_instr'])
    self.active_warps_per_block = float(config['active_warps_per_block'])

    self.cpi = float(config['cpi'])

    self.global_sync = float(config['global_sync'])

    self.num_load = float(config['num_load'])
    self.num_store = float(config['num_store'])

    self.ops_per_point = float(config['ops_per_point'])

    self.shared_latency = float(config['shared_latency'])

    self.num_sm = float(config['num_sm'])


global global_data



def main():

  global global_data

  handle = open(sys.argv[1])
  global_data = GlobalData(yaml.load(handle))
  handle.close()

  gld_issue = global_data.num_load * global_data.cpi
  gld_time = gld_issue + global_data.mem_latency

  # How much latency can we cover?
  gld_hidden_latency = gld_issue * (global_data.active_warps_per_block-1)

  # What is the stall amount?
  gld_stall = max(gld_time - gld_hidden_latency, 0)
  gld_time = gld_time + gld_stall

  # Estimate compute time
  #compute_issue = global_data.ops_per_point * global_data.cpi
  compute_time = global_data.cpi * 24  # Compute latency
  compute_hidden_latency = global_data.cpi * (global_data.active_warps_per_block-1)
  compute_stall = max(0, compute_time - compute_hidden_latency)
  compute = (global_data.cpi + compute_stall) * global_data.ops_per_point

  # Estimate shared-st time
  sst_issue = global_data.num_store * global_data.cpi
  sst_time = sst_issue #+ global_data.shared_latency


  # Estimate shared-ld time
  sld_issue = global_data.num_load * global_data.cpi
  sld_time = sst_issue + global_data.shared_latency

  # How much latency can we cover?
  sld_hidden_latency = sld_issue * (global_data.active_warps_per_block-1)

  # What is the stall amount?
  sld_stall = max(0, sld_time - sld_hidden_latency)
  sld_time = sld_time + sld_stall

  # What is the throughput constraint?
  # It takes 2 cycles to service a load
  sld_throughput_cycles = 2 * global_data.num_load * global_data.active_warps_per_block

  sld_time = max(sld_time, sld_throughput_cycles)


  # Estimate global-st time
  gst_issue = global_data.num_store * global_data.cpi
  gst_time = gst_issue + global_data.mem_latency  # Is this right?


  time = gld_time + compute + sst_time + (global_data.time_tile_size-1)*(sld_time + compute + sst_time) + gst_time

  time = time * global_data.active_warps_per_block

  # Cycles is for one block, scale to all blocks on an SM
  time = time * (global_data.trapezoid_per_stage / global_data.num_sm)

  total = time + global_data.global_sync

  print('Cycles/Iteration: %f' % total)

if __name__ == '__main__':
  main()

