#!/usr/bin/env python

import math
import sys
import yaml


log = True

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
  num_sm = float(config['num_sm'])
  bandwidth = float(config['bandwidth'])
  clock = float(config['clock'])
  elems_per_thread = float(config['elems_per_thread'])



  # What is the per-SM memory bandwidth (approximately)
  bw_per_sm = bandwidth / num_sm

  # What is that in words (or per load)
  words_per_sec = bw_per_sm / 4.0
  print('words_per_sec: %f' % words_per_sec)

  # What is that in words / clock
  words_per_clock = words_per_sec / clock
  print('words_per_clock: %f' % words_per_clock)



  # How long does it take to issue the global loads?
  gld_issue = num_load * cpi * active_warps
  print('gld_issue: %f' % gld_issue)

  # When will we get the first result back?
  gld_first_time = num_load * cpi + mem_latency
  print('gld_first_time: %f' % gld_first_time)

  # Will we stall?
  gld_stall = max(0, gld_first_time - gld_issue)
  print('gld_stall: %f' % gld_stall)

  # Figure out how long until compute can start in the first warp
  gld_time = gld_first_time #+ gld_stall
  print('gld_time: %f' % gld_time)

  # How long will it take to transfer all of the needed data?
  gld_words = num_load * active_warps * 32

  gld_bw_time = math.ceil(gld_words / words_per_clock)
  print('gld_bw_time: %f' % gld_bw_time)

  # Which wins out, latency or bw?
  gld_time = max(gld_time, gld_bw_time)


  # How long does it take to issue a compute instruction for all warps?
  compute_issue = cpi * active_warps
  print('compute_issue: %f' % compute_issue)

  # Can we hide the latency?
  compute_stall = max(0, compute_latency - compute_issue)
  print('compute_stall: %f' % compute_stall)

  # What is the total compute time?
  #compute = (compute_issue + active_warps*compute_stall) * ops_per_point
  compute = compute_issue * ops_per_point + compute_stall * ops_per_point + compute_latency
  print('compute: %f' % compute)

  # Estimate shared-st time
  sst_issue = num_store * cpi * active_warps
  print('sst_issue: %f' % sst_issue)

  # Assume no wait on write
  sst_time = sst_issue
  print('sst_time: %f' % sst_time)

  # Estimate shared-ld time
  sld_issue = num_load * cpi * active_warps
  print('sld_issue: %f' % sld_issue)

  sld_first_time = num_load * cpi + shared_latency
  print('sld_first_time: %f' % sld_first_time)

  # Will we stall?
  sld_stall = max(0, sld_first_time - sld_issue)
  print('std_stall: %f' % sld_stall)

  # When can we start the first compute warp?
  sld_time = sld_first_time + sld_stall
  print('sld_time: %f' % sld_time)

  # What is the throughput constraint?
  # It takes 2 cycles to service a load
  sld_throughput_cycles = (1.0/shared_throughput) * num_load * active_warps
  print('sld_throughput_cycles: %f' % sld_throughput_cycles)

  # Account for bandwidth
  sld_time = max(sld_time, sld_throughput_cycles)
  print('sld_time: %f' % sld_time)

  # Estimate global-st time
  gst_issue = num_store * cpi * active_warps
  print('gst_issue: %f' % gst_issue)

  gst_time = gst_issue #+ mem_latency  # Is this right?
  print('gst_time: %f' % gst_time)

  time = gld_time + compute + sst_time + (time_tile_size-1)*(sld_time + compute + sst_time) + gst_time
  print('cycles (1 point): %f' % time)

  # Account for spatial tiling
  time = time * elems_per_thread
  print('cycles (E points): %f' % time)

  time = time * stages_per_sm
  print('stages_per_sm: %f' % stages_per_sm)
  print('cycles (w/o launch penalty): %f' % time)

  total = time + global_sync

  print('Cycles/Iteration: %f' % total)

if __name__ == '__main__':
  main()

