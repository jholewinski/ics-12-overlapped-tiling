#!/usr/bin/env python

import math
from benchrunner.benchmarks import run_benchmark
import sys
import yaml


if len(sys.argv) != 3:
    print('Usage: %s <arch> <prog>\n' % sys.argv[0])
    exit(1)

arch = sys.argv[1]
prog = sys.argv[2]


if prog == 'j2d':
    # Jacobi 2D Input
    phase2_global_loads = 5.0
    phase2_shared_loads = 0.0
    compute_per_point = 5.0
    phase3_shared_loads = 5.0
    phase4_global_stores = 1.0
    shared_stores = 1.0
    num_fields = 1.0
    data_size = 4.0
    executable = '../../build.out/ocl-jacobi-2d'
    dim = 2
elif prog == 'j1d':
    # Jacobi 1D Input
    phase2_global_loads = 3.0
    phase2_shared_loads = 0.0
    compute_per_point = 3.0
    phase3_shared_loads = 3.0
    phase4_global_stores = 1.0
    shared_stores = 1.0
    num_fields = 1.0
    data_size = 4.0
    executable = '../../build.out/ocl-jacobi-1d'
    dim = 1
elif prog == 'p2d':
    # Poisson 2D Input
    phase2_global_loads = 9.0
    phase2_shared_loads = 0.0
    compute_per_point = 9.0
    phase3_shared_loads = 9.0
    phase4_global_stores = 1.0
    shared_stores = 1.0
    num_fields = 1.0
    data_size = 4.0
    executable = '../../build.out/ocl-poisson-2d'
    dim = 2
elif prog == 'tv2d':
    # CDSC TV Update 2D Input
    phase2_global_loads = 7.0
    phase2_shared_loads = 0.0
    compute_per_point = 59.0
    phase3_shared_loads = 7.0
    phase4_global_stores = 1.0
    shared_stores = 1.0
    num_fields = 1.0
    data_size = 4.0
    executable = '../../build.out/ocl-cdsc-tv-update-2d'
    dim = 2
elif prog == 'g2d':
    # Gradient 2D
    phase2_global_loads = 5.0
    phase2_shared_loads = 0.0
    compute_per_point = 12.0
    phase3_shared_loads = 5.0
    phase4_global_stores = 1.0
    shared_stores = 1.0
    num_fields = 1.0
    data_size = 4.0
    executable = '../../build.out/ocl-gradient-2d'
    dim = 2
elif prog == 'fdtd2d':
    # FDTD 2D Input
    phase2_global_loads = 7.0
    phase2_shared_loads = 4.0
    compute_per_point = 11.0
    phase3_shared_loads = 11.0  
    phase4_global_stores = 3.0
    shared_stores = 3.0
    num_fields = 3.0
    data_size = 4.0
    executable = '../../build.out/ocl-fdtd-2d'
    dim = 2
elif prog == 'rician2d':
    # Rician 2D
    phase2_global_loads = 9.0
    phase2_shared_loads = 0.0
    compute_per_point = 42.0
    phase3_shared_loads = 9.0
    phase4_global_stores = 1.0
    shared_stores = 1.0
    num_fields = 1.0
    data_size = 4.0
    executable = '../../build.out/ocl-rician-denoise-2d'
    dim = 2
else:
    print('Unknown program')
    exit(1)



# Machine Specs
if arch == 'fermi':
    total_shared = 49152.0
    max_warps_per_sm = 48.0
    max_threads_per_sm = 1536.0
    max_blocks_per_sm = 8.0
    num_sm = 14.0
    gmem_bandwidth = 120e9
    smem_bandwidth = 147.2e9
    cpi = 1.0
    clock = 1.15e9

    c_load = 2.0
    c_op = 1.0
    B_gmem = 18.0
    B_smem = 3.0
    L_gmem = 450.0
    L_smem = 32.0

    launch_penalty = 1000

    warp_size = 32.0

elif arch == 'gt200':
    total_shared = 16384.0
    max_warps_per_sm = 32.0
    max_threads_per_sm = 1024.0
    max_blocks_per_sm = 8.0
    num_sm = 30.0
    gmem_bandwidth = 144e9
    smem_bandwidth = 147.2e9
    cpi = 4.0
    clock = 1.30e9

    c_load = 4.0
    c_op = 4.0
    B_gmem = 40.0
    B_smem = 16.0
    L_gmem = 750.0
    L_smem = 32.0

    launch_penalty = 2000

    warp_size = 32.0

elif arch == 'gcn':
    total_shared = 32768.0
    max_warps_per_sm = 16.0
    max_threads_per_sm = 1024.0
    max_blocks_per_sm = 8.0
    num_sm = 32.0
    gmem_bandwidth = 144e9
    smem_bandwidth = 147.2e9
    cpi = 4.0
    clock = 1.30e9

    c_load = 4.0
    c_op = 4.0
    B_gmem = 16.0
    B_smem = 8.0
    L_gmem = 550.0
    L_smem = 32.0

    launch_penalty = 2000

    warp_size = 64.0


# Configuration Space

#block_size_x = [64]
#block_size_y = [6]
#block_size_z = [1]
#time_tile_size = [1, 2, 3, 4]
#elems_per_thread = [4, 6, 8]


if dim == 1:
    block_size_x = [128, 256, 512, 1024]
    block_size_y = [1]
    block_size_z = [1]
elif dim == 2:
    block_size_x = [32, 64]
    block_size_y = [4, 6]
    block_size_z = [1]
else:
    block_size_x = [8, 16, 32]
    block_size_y = [2, 4, 6]
    block_size_z = [2, 4, 6]


time_tile_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elems_per_thread = [2, 4, 6, 8]

phases = [0]




# Build configuration space
configs = []
for x in block_size_x:
    for y in block_size_y:
        for z in block_size_z:
            for t in time_tile_size:
                for e in elems_per_thread:
                    for p in phases:
                        configs.append([x, y, z, t, e, p])


# Output Headers
headers = ['x', 'y', 'z', 't', 'e', 'phase_limit', 'real_elapsed', 'real_per_stage', 'active_warps', 'full_invocations', 'extra_invocations', 'extra_global', 'extra_shared', 'num_stages', 't_glb', 't_shd', 't_stage', 't_stage_extra', 'sim_elapsed', 'sim_elapsed_clk', 'pts_per_clk']

# Print header
for h in headers:
    sys.stdout.write('%s,' % h)
sys.stdout.write('\n')
sys.stdout.flush()


min_real = 100000.0

results = []

avg_error = 0.0
min_error = 100000.0
max_error = -100000.0

# Iterate configuration space
for (x, y, z, t, e, phase_limit) in configs:

    if dim == 1:
        problem_size = 5000
    elif dim == 2:
        problem_size = 2000
    else:
        problem_size = 64

    time_steps = 64


    """
    Real Stats
    """
    (ret, stdout, stderr, run_time) = run_benchmark(executable, problem_size, time_steps, t, e, bsx=x, bsy=y, bsz=z, phase=phase_limit)
    if ret == 0:
        doc = yaml.load(stdout)
        real_elapsed = float(doc['Elapsed Time'])
        num_blocks_x = float(doc['Num Blocks X'])
        try:
            num_blocks_y = float(doc['Num Blocks Y'])
        except:
            num_blocks_y = 1.0
        try:
            num_blocks_z = float(doc['Num Blocks Z'])
        except:
            num_blocks_z = 1.0
    else:
        real_elapsed = 1e10
        sys.stderr.write(stderr)
        sys.stderr.flush()
        continue


    """
    Derived Stats
    """
    warps_per_block = x*y*z/warp_size
    shared_per_block = (x+2.0) * (y+2.0) * (z+2.0) * 4.0  # ESTIMATE
    blocks_per_sm = min(
        math.floor(max_warps_per_sm / warps_per_block),
        math.floor(total_shared / shared_per_block))
    blocks_per_sm = max(min(blocks_per_sm, max_blocks_per_sm), 1.0)

    active_warps = warps_per_block * blocks_per_sm

    real_per_block_x = x - 2.0 * (t-1.0) # ESTIMATE
    if dim > 1:
        real_per_block_y = y * e - 2.0 * (t-1.0)
    else:
        real_per_block_y = 1.0
    if dim > 2:
        real_per_block_z = z - 2.0 * (t-1.0)
    else:
        real_per_block_z = 1.0

    real_per_block = real_per_block_x * real_per_block_y * real_per_block_z
    real_per_stage = real_per_block * blocks_per_sm     # Remember, this is *per* SM

    total_blocks = num_blocks_x * num_blocks_y * num_blocks_z
    num_stages = math.ceil(total_blocks / (blocks_per_sm * num_sm))

    full_invocations = int(time_steps) / int(t)
    extra_invocations = int(time_steps) % int(t)
    if extra_invocations > 0:
        extra_global = 1
    else:
        extra_global = 0
    extra_shared = max(0, extra_invocations-1)


    """
    Performance Estimation
    """    

    # Cache Model
    if arch == 'fermi' or arch == 'gcn':
        p2_glb = 5.0*e
        p2_glb = e+4.0
        p2_shd = 5.0*e - p2_glb
    else:
        # No Cache
        p2_glb = phase2_global_loads * e
        p2_shd = phase2_shared_loads * e

    k_op = compute_per_point * e

    t_glb = max(c_load*(p2_glb+p2_shd)*active_warps + c_op*k_op*active_warps,
                L_gmem + c_load + max(B_gmem*p2_glb*active_warps, B_smem*p2_shd*e*active_warps))

    #t_glb = max(c_load*(p2_glb+p2_shd)*active_warps + c_op*k_op*active_warps,
    #            L_gmem + c_load + B_gmem*p2_glb*active_warps)

    t_shd = max(c_load*phase3_shared_loads*e*active_warps + c_op*k_op*active_warps,
                L_smem + c_load + B_smem *phase3_shared_loads*e*active_warps)

    t_phase1 = c_op*20*active_warps

    if phase_limit == 1:
        t_stage = 0.0
        t_stage_extra = 0.0
    elif phase_limit == 2:
        t_stage = t_glb
        t_stage_extra = t_glb
    elif phase_limit == 3:
        t_stage = (t-1)*t_shd
        t_stage_extra = (extra_shared)*t_shd
    else:
        t_stage = t_glb + (t-1)*t_shd
        t_stage_extra = (extra_global*t_glb) + (extra_shared*t_shd)

    t_stage = t_stage + t_phase1
    if extra_global > 0:
        t_stage_extra = t_stage_extra + t_phase1
    
    t_stage_total = t_stage + t_stage_extra


    #sim_elapsed = (t_stage + invocations*launch_penalty) / clock * num_stages * full_invocations

    sim_elapsed_clk = ((t_stage * num_stages) + full_invocations*launch_penalty) * (int(time_steps) / int(t))
    if extra_global > 0:
        sim_elapsed_clk = sim_elapsed_clk + ((t_stage_extra * num_stages) + launch_penalty)
    sim_elapsed = sim_elapsed_clk / clock


    if t_stage > 0.0:
        #pts_per_clk = (real_per_stage*t) / t_stage
        pts_per_clk = (real_per_block_x * real_per_block_y * real_per_block_z * num_blocks_x * num_blocks_y * num_blocks_z * time_steps) / sim_elapsed_clk
    else:
        pts_per_clk = 0.0

    min_real = min(min_real, real_elapsed)

    error = (sim_elapsed - real_elapsed) / real_elapsed
    avg_error = avg_error + error

    min_error = min(min_error, error)
    max_error = max(max_error, error)

    results.append([x, y, z, t, e, real_elapsed, sim_elapsed, pts_per_clk])

    """
    Print Results
    """
    for h in headers:
        sys.stdout.write('%s,' % str(locals()[h]))
    sys.stdout.write('\n')
    sys.stdout.flush()


# Find best results
sorted_results = sorted(results, key=lambda x: x[7])
sorted_results.reverse()


for pt in sorted_results[:3]:
    (x, y, z, t, e, real, sim, pts_per_clk) = pt
    rel_time = real / min_real
    sys.stderr.write('%s: %f\n' % (str(pt), rel_time))
    sys.stderr.flush()

# Print stats
avg_error = avg_error / float(len(results))

sys.stderr.write('# Average Error: %f\n' % avg_error)
sys.stderr.write('# Maximum Error: %f\n' % max_error)
sys.stderr.write('# Minimum Error: %f\n' % min_error)


