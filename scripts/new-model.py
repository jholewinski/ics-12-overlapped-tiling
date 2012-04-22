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
  gld_stall = gld_time - gld_hidden_latency

  # Estimate compute time
  compute = global_data.ops_per_point * global_data.cpi

  # Estimate shared-st time
  sst_issue = global_data.num_store * global_data.cpi
  sst_time = sst_issue + global_data.shared_latency


  # Estimate shared-ld time
  sld_issue = global_data.num_load * global_data.cpi
  sld_time = sst_issue + global_data.shared_latency

  # How much latency can we cover?
  sld_hidden_latency = sld_issue * (global_data.active_warps_per_block-1)

  # What is the stall amount?
  sld_stall = gld_time - gld_hidden_latency


  # Estimate global-st time
  gst_issue = global_data.num_store * global_data.cpi
  gst_time = gst_issue + global_data.mem_latency  # Is this right?


  time = gld_time + compute + sst_time + (global_data.time_tile_size-1)*(sld_time + compute + sst_time) + gst_time

  # Cycles is for one block, scale to all blocks on an SM
  time = time * (global_data.trapezoid_per_stage / global_data.num_sm)

  total = time + global_data.global_sync

  print('Cycles/Iteration: %f' % total)

if __name__ == '__main__':
  main()

