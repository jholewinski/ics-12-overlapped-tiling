#!/usr/bin/env python

from math import *
from subprocess import *
import os.path
import sys
import math

#program = '../../build.out/ocl-cdsc-tv-update-2d'
program = './ocl-jacobi-2d'

#time_tile_sizes = [2, 3, 4, 5, 6, 7, 8]
#elems_per_thread = [10]

time_tile_sizes = [2, 4]
elems_per_thread = [6, 8]

#block_x = range(32, 64+1, 16)
#block_y = range(8, 16+1, 4)

block_x = [16, 32, 48, 64]
block_y = [8, 16]

#block_x = [32, 48]
#block_y = [8, 16]

sim_program = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new-model-2.py')

data_points = []
min_elapsed = 1000.0

log = open('run.log', 'w')

num_runs = 3

phase_limit = 0

print('bsx,bsy,tts,elems,elapsed,variance,sim,gflops,occupancy,')
for tts in time_tile_sizes:
  for elems in elems_per_thread:
    for bsx in block_x:
      for bsy in block_y:
        log.write('======== Running S=%d E=%d\n' % (tts, elems))
        args = '%s -x %d -y %d -w kernel.tmp.cl -n 1024 -t 64 -s %d -e %d -p %d' % (program, bsx, bsy, tts, elems, phase_limit)

        elapsed_times = []

        for n in range(0, num_runs):
          proc = Popen(args.split(), stdout=PIPE, stderr=PIPE)
          (outs, errs) = proc.communicate()
          log.write(outs)
          log.write(errs)
          if proc.returncode != 0:
            continue

          outs = outs.strip()
          lines = outs.split('\n')
          program_data = {}
          for l in lines:
            data = l.split(':')
            program_data[data[0].strip()] = data[1].strip()

          elapsed_times.append(float(program_data['Elapsed Time']))


        expect_xsq = 0.0
        expect_x = 0.0

        for t in elapsed_times:
          expect_xsq = expect_xsq + t * t
          expect_x = expect_x + t

        expect_xsq = expect_xsq / float(len(elapsed_times))
        expect_x = expect_x / float(len(elapsed_times))
        variance = expect_xsq - expect_x * expect_x
        elapsed_time = expect_x


        # Write YAML for simulator
        file_handle = open('tmp.yaml', 'w')

        file_handle.write('phase_limit: %d\n' % phase_limit)


        # J2D
        file_handle.write('elems_per_op: 5\n')
        file_handle.write('num_arrays: 1\n')
        file_handle.write('num_load: 5\n')
        file_handle.write('num_store: 1\n')
        file_handle.write('ops_per_point: 5\n')

        # P2D
        #file_handle.write('elems_per_op: 9\n')
        #file_handle.write('num_arrays: 1\n')
        #file_handle.write('num_load: 9\n')
        #file_handle.write('num_store: 1\n')
        #file_handle.write('ops_per_point: 9\n')

        # TV Update 2D
        #file_handle.write('elems_per_op: 7\n')
        #file_handle.write('num_arrays: 1\n')
        #file_handle.write('num_load: 7\n')
        #file_handle.write('num_store: 1\n')
        #file_handle.write('ops_per_point: 57\n')

        # Rician 2D
        #file_handle.write('elems_per_op: 9\n')
        #file_handle.write('num_arrays: 1\n')
        #file_handle.write('num_load: 5\n')
        #file_handle.write('num_store: 1\n')
        #file_handle.write('ops_per_point: 42\n')

        # Gradient 2D
        #file_handle.write('elems_per_op: 5\n')
        #file_handle.write('num_arrays: 1\n')
        #file_handle.write('num_load: 5\n')
        #file_handle.write('num_store: 1\n')
        #file_handle.write('ops_per_point: 15\n')

        # FDTD 2D
        #file_handle.write('elems_per_op: 7\n')
        #file_handle.write('num_arrays: 3\n')
        #file_handle.write('num_load: 7\n')
        #file_handle.write('num_store: 3\n')
        #file_handle.write('ops_per_point: 11\n')


        # Arch parameters
        total_shared_per_sm = 49152.0
        max_warps_per_sm = 48.0
        max_threads_per_sm = 1536.0
        max_blocks_per_sm = 8.0
        num_sm = 14.0


        file_handle.write('num_sm: %d\n' % num_sm)
        file_handle.write('mem_latency: 500\n')
        file_handle.write('bandwidth: 120e9\n')
        file_handle.write('cpi: 1\n')
        file_handle.write('global_sync: 3350\n')
        file_handle.write('shared_latency: 32\n')
        file_handle.write('compute_latency: 24\n')
        file_handle.write('shared_throughput: 0.5\n')
        file_handle.write('clock: 1.15e9\n')


        # Program properties
        block_size_x = float(program_data['Block Size X'])
        block_size_y = float(program_data['Block Size Y'])
        num_warps_per_block = block_size_x * block_size_y / 32.0
        shared_size_per_block = float(program_data['Shared Size'])
        num_blocks_x = float(program_data['Num Blocks X'])
        num_blocks_y = float(program_data['Num Blocks Y'])

        # Figure out how many blocks we can execute concurrently
        blocks_per_sm = min(
          math.floor(max_warps_per_sm / num_warps_per_block),
          math.floor(total_shared_per_sm / max(1, shared_size_per_block)))
        blocks_per_sm = max(blocks_per_sm, 1.0)

        # Now how many warps is that?
        active_warps_per_sm = num_warps_per_block * blocks_per_sm

        file_handle.write('active_warps: %f\n' % active_warps_per_sm)

        total_blocks_per_sm = math.ceil(num_blocks_x * num_blocks_y / num_sm)
        stages_per_sm = math.ceil(total_blocks_per_sm / blocks_per_sm)
        file_handle.write('stages_per_sm: %f\n' % stages_per_sm)

        file_handle.write('time_tile_size: %d\n' % tts)
        file_handle.write('elems_per_thread: %d\n' % elems)

        #concurr_blocks_per_sm = floor(min(2.0, total_shared_per_sm / shared_size))
        #concurr_blocks_per_sm = max(concurr_blocks_per_sm, 1)
        #concurr_blocks_per_sm = 8.0
        #file_handle.write('active_warps_per_block: %f\n' % num_warps)
        #file_handle.write('concurrent_blocks: %f\n' % concurr_blocks_per_sm)
        #file_handle.write('num_sm: 14\n')

        actual_gflops = float(program_data['Actual GFlop/s'])

        elems_per_block_x = block_size_x
        elems_per_block_y = block_size_y * elems
        file_handle.write('elems_per_block_x: %d\n' % elems_per_block_x)
        file_handle.write('elems_per_block_y: %d\n' % elems_per_block_y)

        file_handle.write('real_per_block_x: %d\n' % int(program_data['Real Per Block X']))
        file_handle.write('real_per_block_y: %d\n' % int(program_data['Real Per Block Y']))



        #num_blocks = num_blocks_x * num_blocks_y
        #file_handle.write('trapezoid_per_stage: %f\n' % num_blocks)
        #file_handle.write('concurr_blocks_per_sm: %f\n' % concurr_blocks_per_sm)

        #num_invocations = 64.0 / float(tts)
        #file_handle.write('num_invocations: %f\n' % num_invocations)

        #file_handle.write('clock: 1.30\n')
        #file_handle.write('kernel_launch_penalty: 5000\n')
        file_handle.close()

        # Run the sim
        args = '%s tmp.yaml' % sim_program
        proc = Popen(args.split(), stdout=PIPE, stderr=PIPE)
        (outs, errs) = proc.communicate()
        log.write(errs)
        log.write(outs)
        if proc.returncode != 0:
          print(errs)
          print('Sim failed?!?')
          exit(1)
        outs = outs.strip()
        lines = outs.split('\n')
        sim_data = {}
        for l in lines:
          data = l.split(':')
          sim_data[data[0].strip()] = data[1].strip()

        sim_cycles = float(sim_data['Cycles/Iteration'])

        sim_time = sim_cycles / 1.15e9 * (64.0 / float(tts))

        log.write('Elapsed: %f  Simulated: %f\n' % (elapsed_time, sim_time))

        data_points.append([bsx, bsy, tts, elems, elapsed_time, sim_time])
        min_elapsed = min(min_elapsed, elapsed_time)

        occupancy = active_warps_per_sm / max_warps_per_sm

        print('%d,%d,%d,%d,%f,%E,%f,%f,%f,' % (bsx, bsy, tts, elems, elapsed_time, variance, sim_time, actual_gflops, occupancy))
        sys.stdout.flush()

#sorted_data = sorted(data_points, key=lambda pt: pt[3])
#sorted_data = data_points

#print('Min Elapsed: %f' % min_elapsed)
#for pt in sorted_data:
#  print('S: %d - E: %d - Elapsed: %f - Sim: %f' % (pt[0], pt[1], pt[2], pt[3]))


