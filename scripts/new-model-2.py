#!/usr/bin/env python

import sys
import yaml




def main():

  handle = open(sys.argv[1])
  config = yaml.load(handle)
  handle.close()


  num_load = float(config['num_load'])
  num_store = float(config['num_store'])
  ops_per_point = float(config['ops_per_point'])
  cpi = float(config['cpi'])
  mem_latency = float(config['mem_latency'])
  active_warps = float(config['active_warps'])
  compute_latency = float(config['compute_latency'])
  shared_latency = float(config['shared_latency'])
  shared_throughput = float(config['shared_throughput'])
  time_tile_size = float(config['time_tile_size'])
  stages_per_sm = float(config['stages_per_sm'])
  global_sync = float(config['global_sync'])

  # How long does it take to issue the global loads?
  gld_issue = num_load * cpi * active_warps
  # When will we get the first result back?
  gld_first_time = num_load * cpi + mem_latency

  # Will we stall?
  gld_stall = max(0, gld_first_time - gld_issue)

  # Figure out how long until compute can start in the first warp
  gld_time = gld_first_time + gld_stall

  # How long does it take to issue a compute instruction for all warps?
  compute_issue = cpi * active_warps

  # Can we hide the latency?
  compute_stall = max(0, compute_latency - compute_issue)

  # What is the total compute time?
  compute = (compute_issue + active_warps*compute_stall) * ops_per_point

  # Estimate shared-st time
  sst_issue = num_store * cpi * active_warps

  # Assume no wait on write
  sst_time = sst_issue


  # Estimate shared-ld time
  sld_issue = num_load * cpi * active_warps

  sld_first_time = num_load * cpi + shared_latency

  # Will we stall?
  sld_stall = max(0, sld_first_time - sld_issue)

  # When can we start the first compute warp?
  sld_time = sld_first_time + sld_stall

  # What is the throughput constraint?
  # It takes 2 cycles to service a load
  sld_throughput_cycles = (1.0/shared_throughput) * num_load * active_warps

  # Account for bandwidth
  sld_time = max(sld_time, sld_throughput_cycles)


  # Estimate global-st time
  gst_issue = num_store * cpi * active_warps
  gst_time = gst_issue + mem_latency  # Is this right?


  time = gld_time + compute + sst_time + (time_tile_size-1)*(sld_time + compute + sst_time) + gst_time

  time = time * stages_per_sm

  total = time + global_sync

  print('Cycles/Iteration: %f' % total)

if __name__ == '__main__':
  main()

