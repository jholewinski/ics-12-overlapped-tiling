#!/usr/bin/env python

import math
import sys


if len(sys.argv) != 3:
    print('Usage: %s <prog_data> <arch_file>\n' % sys.argv[0])
    exit(1)


# Source the architecture config file
with open(sys.argv[2]) as arch:
    arch_globals = globals()
    exec arch in arch_globals

arch = 'gt200'

# Read the run data
all_runs = []
with open(sys.argv[1]) as data:
    headers = data.readline()
    headers = headers.strip().split(',')[:-1]
    for line in data.readlines():
        comps = line.strip().split(',')[:-1]
        run = zip(headers, comps)
        run = dict(run)
        all_runs.append(run)




# Output Headers
headers = ['count', 'x', 'y', 'z', 't', 'e', 'phase_limit', 'real_elapsed', 'event_elapsed', 'real_per_stage', 'active_warps', 'blocks_from_max_warps', 'blocks_from_max_shared', 'blocks_from_max_regs', 'shared_per_block', 'blocks_per_sm', 'num_blocks_x', 'num_blocks_y', 'num_blocks_z', 'full_invocations', 'extra_invocations', 'extra_global', 'extra_shared', 'num_stages', 't_glb', 't_shd', 't_phase1', 't_stage', 't_stage_extra', 'sim_elapsed', 'sim_elapsed_clk', 'sim_elapsed_upper', 'sim_elapsed_upper_clk', 'glb_factor', 'shd_factor', 'pts_per_clk', 'badness']

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


count = 1

# Iterate configuration space
for run in all_runs:

    x = int(run['Block Size X'])
    y = int(run['Block Size Y'])
    z = int(run['Block Size Z'])
    e = int(run['Elements/Thread'])
    t = int(run['Time Tile Size'])


    dim = int(run['Dimensions'])

    size_ratio = float(x) / float(y*e)
    if dim == 3:
        size_ratio_xz = float(x) / float(z)
        size_ratio_yz = float(y*e) / float(z)
        size_ratio = (size_ratio + size_ratio_xz + size_ratio_yz) / 3.0

    badness = abs(size_ratio - 1.0)

    if badness > 1.0:
        continue

    regs_per_thread = int(run['Register Usage'])

    num_blocks_x = int(run['Num Blocks X'])
    num_blocks_y = int(run['Num Blocks Y'])
    num_blocks_z = int(run['Num Blocks Z'])

    time_steps = int(run['Time Steps'])

    phase2_global_loads = int(run['phase2_global_loads'])
    phase2_shared_loads = int(run['phase2_shared_loads'])
    compute_per_point = int(run['compute_per_point'])
    phase3_shared_loads = int(run['phase3_shared_loads'])
    phase4_global_stores = int(run['phase4_global_stores'])
    shared_stores = int(run['shared_stores'])
    num_fields = int(run['num_fields'])
    data_size = int(run['data_size'])

    phase_limit = int(run['phase_limit'])

    real_elapsed = float(run['Elapsed Time'])

    event_elapsed = float(run['EventElapsed'])
    
    """
    Derived Stats
    """
    warps_per_block = x*y*z/warp_size

    if dim == 1:
        shared_per_block = (e*x+2.0) * 4.0
    elif dim == 2:
        shared_per_block = (x+2.0) * (e*y+2.0) * 4.0
    elif dim == 3:
        shared_per_block = (x+2.0) * (e*y+2.0) * (z+2.0) * 4.0  # ESTIMATE
    else:
        print('Invalid dim')
        exit(1)

    regs_per_block = regs_per_thread*x*y*z

    blocks_from_max_warps = math.floor(max_warps_per_sm / warps_per_block)
    blocks_from_max_shared = math.floor(total_shared / shared_per_block)
    blocks_from_max_regs = math.floor(max_regs_per_sm / regs_per_block)

    total_blocks = num_blocks_x * num_blocks_y * num_blocks_z

    blocks_per_sm = min(
        blocks_from_max_warps,
        blocks_from_max_shared,
        blocks_from_max_regs,
        #math.floor(total_blocks/num_sm))
        math.ceil(total_blocks/num_sm))
    blocks_per_sm = max(min(blocks_per_sm, max_blocks_per_sm), 1.0)


    #extra_blocks = total_blocks % num_sm

    #extra_blocks_bw = B_gmem * (1.0 / (num_sm / extra_blocks))

    #active_warps_extra = warps_per_block



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



    num_stages = (total_blocks / (blocks_per_sm * num_sm)) # CEIL

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
    if cache_model == 'fermi':
        # if prog == 'j2d':
        #     p2_glb = 5.0*e
        #     p2_glb = e+4.0
        # elif prog == 'fdtd2d':
        #     e_min = elems_per_thread[0]
        #     e_max = elems_per_thread[-1]

        #     t1 = e_max - e_min
        #     t2 = e_max - e
        #     t3 = t2 / t1
        #     t4 = t3 / 0.5
        #     t5 = 1.0 - t4
            

        #     p2_glb = phase2_global_loads * e
        #     p2_glb = 4.0 * e + 1.0

        # elif prog == 'g2d':
        #     p2_glb = e + 4.0
        # elif prog == 'p2d':
        #     p2_glb = 3*e + 6.0
        # elif prog == 'rician2d':
        #     p2_glb = e + 4.0
        # elif prog == 'j3d':
        #     p2_glb = 3.0*e + 4.0
        # elif prog == 'tv2d':
        #     p2_glb = 2.0*e + 5.0
        # elif prog == 'j1d':
        #     p2_glb = phase2_global_loads * e
        # else:
        #     sys.stderr.write('No cache model!\n')
        #     exit(1)

        #if dim == 3 and phase2_global_loads == 7.0:
        #    p2_glb = 3.0*e + 4.0
        #else:
        #    sys.stderr.write('Dont know how to apply cache model\n')
        #    sys.exit(1)

        misses = float(run['l1_global_load_miss'])
        hits = float(run['l1_global_load_hit'])

        miss_rate = misses / (misses + hits)

        p2_glb = phase2_global_loads*e * miss_rate
        p2_shd = phase2_shared_loads*e + phase2_global_loads*e - p2_glb

        #p2_glb = p2_glb * ((2.0 ** (1.0/8.0)) ** e)
        #p2_shd = phase2_global_loads*e - p2_glb
        #p2_shd = p2_shd + phase2_shared_loads*e

    else:
        # No Cache
        p2_glb = phase2_global_loads * e
        p2_shd = phase2_shared_loads * e

    k_op = compute_per_point * e

    #Bprime = 8.0*p2_glb + B_gmem
    Bprime = B_gmem

    #t_glb = max(c_load*(p2_glb+p2_shd)*active_warps + c_op*k_op*active_warps + c_load*(p2_glb+p2_shd)*active_warps_extra + c_op*k_op*active_warps_extra,
    #            L_gmem + c_load + max(Bprime*p2_glb*active_warps + extra_blocks_bw*p2_glb*active_warps_extra, B_smem*p2_shd*active_warps + B_smem*p2_shd*active_warps_extra))
    t_glb = max(c_load*(p2_glb+p2_shd)*active_warps + c_op*k_op*active_warps,
                L_gmem + c_load + max(Bprime*p2_glb*active_warps, B_smem*p2_shd*active_warps))
    t_glb_upper = max(0,#c_load*(p2_glb+p2_shd)*active_warps + c_op*k_op*active_warps,
                      L_gmem + c_load + max(Bprime*p2_glb*active_warps, B_smem*p2_shd*active_warps) + c_op*k_op*active_warps + c_load*(p2_glb+p2_shd)*active_warps)



    #t_glb = max(c_load*(p2_glb+p2_shd)*active_warps + c_op*k_op*active_warps,
    #            L_gmem + c_load + B_gmem*p2_glb*active_warps)

    t_shd = max(c_load*phase3_shared_loads*e*active_warps + c_op*k_op*active_warps,
                L_smem + c_load + B_smem *phase3_shared_loads*e*active_warps)
    t_shd_upper = max(0,#c_load*phase3_shared_loads*e*active_warps + c_op*k_op*active_warps,
                      L_smem + c_load + B_smem *phase3_shared_loads*e*active_warps + c_op*k_op*active_warps + c_load*phase3_shared_loads*e*active_warps)

    if c_load*(p2_glb+p2_shd)*active_warps + c_op*k_op*active_warps > L_gmem + c_load + max(Bprime*p2_glb*active_warps, B_smem*p2_shd*active_warps):
        glb_factor = 'latency'
    else:
        glb_factor = 'bandwidth'

    if c_load*phase3_shared_loads*e*active_warps + c_op*k_op*active_warps > L_smem + c_load + B_smem *phase3_shared_loads*e*active_warps:
        shd_factor = 'latency'
    else:
        shd_factor = 'bandwidth'


    write_back = L_gmem + c_load + B_gmem*phase4_global_stores*active_warps

    t_phase1 = (30*e+50)*active_warps

    if phase_limit == 1:
        t_stage = 0.0
        t_stage_extra = 0.0
        t_stage_upper = 0.0
        t_stage_extra_upper = 0.0
    elif phase_limit == 2:
        t_stage = t_glb + write_back
        t_stage_extra = (extra_global*t_glb)
        t_stage_upper = t_glb_upper 
        t_stage_extra_upper = t_glb_upper
    elif phase_limit == 3:
        t_stage = (t-1)*t_shd
        t_stage_extra = (extra_shared)*t_shd
        t_stage_upper = (t-1)*t_shd_upper
        t_stage_extra_upper = (extra_shared)*t_shd_upper
    else:
        t_stage = t_glb + (t-1)*t_shd + write_back
        t_stage_extra = (extra_global*t_glb) + (extra_shared*t_shd) + (extra_global*write_back)
        t_stage_upper = t_glb_upper + (t-1)*t_shd_upper + write_back
        t_stage_extra_upper = (extra_global*t_glb_upper) + (extra_shared*t_shd_upper) + (extra_global*write_back)

    t_stage = t_stage + t_phase1
    t_stage_upper = t_stage_upper + t_phase1

    if extra_global > 0:
        t_stage_extra = t_stage_extra + t_phase1
        t_stage_extra_upper = t_stage_extra_upper + t_phase1
    
    #t_stage_total = t_stage + t_stage_extra


    #sim_elapsed = (t_stage + invocations*launch_penalty) / clock * num_stages * full_invocations

    sim_elapsed_clk = ((t_stage * num_stages) + launch_penalty) * full_invocations
    if extra_global > 0:
        sim_elapsed_clk = sim_elapsed_clk + ((t_stage_extra * num_stages) + launch_penalty)
    sim_elapsed = sim_elapsed_clk / clock


    sim_elapsed_upper_clk = ((t_stage_upper * num_stages) + launch_penalty) * full_invocations
    if extra_global > 0:
        sim_elapsed_upper_clk = sim_elapsed_upper_clk + ((t_stage_extra_upper * num_stages) + launch_penalty)
    sim_elapsed_upper = sim_elapsed_upper_clk / clock


    if t_stage > 0.0:
        #pts_per_clk = (real_per_stage*t) / t_stage
        pts_per_clk = (real_per_block_x * real_per_block_y * real_per_block_z * num_blocks_x * num_blocks_y * num_blocks_z * time_steps) / sim_elapsed_clk
    else:
        pts_per_clk = 0.0

    min_real = min(min_real, event_elapsed)

    error = abs((sim_elapsed - event_elapsed) / event_elapsed)
    avg_error = avg_error + error

    min_error = min(min_error, error)
    max_error = max(max_error, error)

    results.append([x, y, z, t, e, event_elapsed, sim_elapsed, pts_per_clk])

    """
    Print Results
    """
    for h in headers:
        sys.stdout.write('%s,' % str(locals()[h]))
    sys.stdout.write('\n')
    sys.stdout.flush()

    count = count + 1

# Find best results
sorted_results = sorted(results, key=lambda x: x[6])
#sorted_results.reverse()


max_overhead = 0.0
for (_, _, _, _, _, real, _, _) in results:
    overhead = real / min_real
    max_overhead = max(max_overhead, overhead)

for pt in sorted_results[:5]:
    (x, y, z, t, e, real, sim, pts_per_clk) = pt
    rel_time = real / min_real
    sys.stderr.write('%s: %f\n' % (str(pt), rel_time))
    sys.stderr.flush()

# Print stats
avg_error = avg_error / float(len(results))

sys.stderr.write('# Maximum Overhead: %f\n' % max_overhead)
sys.stderr.write('# Average Error:    %f\n' % avg_error)
sys.stderr.write('# Maximum Error:    %f\n' % max_error)
sys.stderr.write('# Minimum Error:    %f\n' % min_error)
sys.stderr.write('# Num Results:      %d (Culled: %d)\n' % (len(results), (len(all_runs) - len(results))))
