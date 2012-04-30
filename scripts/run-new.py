#!/usr/bin/env python

from math import *
from subprocess import *
import os.path

program = '../../build.out/ocl-jacobi-2d'

time_tile_sizes = [1, 2, 3, 4]
elems_per_thread = [4, 6, 8, 10]
block_x = range(16, 64+1, 16)
block_y = range(4, 16+1, 4)

dump_program = './dump-cl-binary.x'
sim_program = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new-model.py')

data_points = []
min_elapsed = 1000.0

log = open('run.log', 'w')

print('bsx, bsy, tts, elems, elapsed, sim, gflops')
for tts in time_tile_sizes:
  for elems in elems_per_thread:
    for bsx in block_x:
      for bsy in block_y:
        log.write('======== Running S=%d E=%d\n' % (tts, elems))
        args = '%s -x %d -y %d -w kernel.tmp.cl -n 1024 -t 64 -s %d -e %d' % (program, bsx, bsy, tts, elems)
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

        elapsed_time = float(program_data['Elapsed Time'])

        # Get PTX
        args = '%s kernel.tmp.cl' % dump_program
        call(args.split())
        file_handle = open('kernel.tmp.cl.ptx')
        ptx = file_handle.read()
        file_handle.close()
        log.write(ptx)

        # Post-process PTX
        ptx = ptx.replace(', texmode_independent', '')
        ptx = ptx.replace('.ptr .global', '')

        # Save it to a temp file
        file_handle = open('kernel.tmp.ptx', 'w')
        file_handle.write(ptx)
        file_handle.close()

        # Write YAML for simulator
        file_handle = open('tmp.yaml', 'w')

        file_handle.write('mem_latency: 300\n')
        file_handle.write('mem_pipeline_depth: 16\n')
        file_handle.write('bandwidth: 141.7e9\n')
        file_handle.write('coalesce_degree: 16\n')
        file_handle.write('data_size: 4\n')
        file_handle.write('cpi: 4\n')
        file_handle.write('global_sync: 3350\n')
        file_handle.write('shared_latency: 4\n')


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


        num_warp_instrs = tts * 20
        file_handle.write('num_warp_instr: %d\n' % num_warp_instrs)

        shared_size = float(program_data['Shared Size'])
        total_shared_per_sm = 16384.0

        file_handle.write('time_tile_size: %d\n' % tts)

        concurr_blocks_per_sm = floor(min(2.0, total_shared_per_sm / shared_size))
        concurr_blocks_per_sm = max(concurr_blocks_per_sm, 1)
        block_size_x = float(program_data['Block Size X'])
        block_size_y = float(program_data['Block Size Y'])
        num_warps = block_size_x * block_size_y / 32.0
        #concurr_blocks_per_sm = 8.0
        file_handle.write('active_warps_per_block: %f\n' % num_warps)
        file_handle.write('concurrent_blocks: %f\n' % concurr_blocks_per_sm)
        file_handle.write('num_sm: 30\n')

        actual_gflops = float(program_data['Actual GFlop/s'])

        elems_per_block_x = block_size_x
        elems_per_block_y = block_size_y * elems
        file_handle.write('elems_per_block_x: %d\n' % elems_per_block_x)
        file_handle.write('elems_per_block_y: %d\n' % elems_per_block_y)

        file_handle.write('real_per_block_x: %d\n' % int(program_data['Real Per Block X']))
        file_handle.write('real_per_block_y: %d\n' % int(program_data['Real Per Block Y']))


        num_blocks_x = float(program_data['Num Blocks X'])
        num_blocks_y = float(program_data['Num Blocks Y'])
        num_blocks = num_blocks_x * num_blocks_y
        file_handle.write('trapezoid_per_stage: %f\n' % num_blocks)
        file_handle.write('concurr_blocks_per_sm: %f\n' % concurr_blocks_per_sm)

        num_invocations = 64.0 / float(tts)
        file_handle.write('num_invocations: %f\n' % num_invocations)

        file_handle.write('clock: 1.30\n')
        file_handle.write('kernel_launch_penalty: 5000\n')
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

        sim_time = sim_cycles / 1.30e9 * (64.0 / float(tts))

        log.write('Elapsed: %f  Simulated: %f\n' % (elapsed_time, sim_time))

        data_points.append([bsx, bsy, tts, elems, elapsed_time, sim_time])
        min_elapsed = min(min_elapsed, elapsed_time)

        print('%d,%d,%d,%d,%f,%f,%f' % (bsx, bsy, tts, elems, elapsed_time, sim_time, actual_gflops))
#sorted_data = sorted(data_points, key=lambda pt: pt[3])
#sorted_data = data_points

#print('Min Elapsed: %f' % min_elapsed)
#for pt in sorted_data:
#  print('S: %d - E: %d - Elapsed: %f - Sim: %f' % (pt[0], pt[1], pt[2], pt[3]))


