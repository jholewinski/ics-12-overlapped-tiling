#
# Support routines for experiment harness
#

import os
import subprocess
import sys
import time

try:
    import yaml
except:
    print('Please install PyYAML')
    sys.exit(1)

def _parse_range(value):
    if isinstance(value, int):
        return [value]
    elif len(value) == 3:
        return range(value[0], value[1]+1, value[2])
    elif len(value) == 2:
        return range(value[0], value[1]+1, 1)
    elif len(value) == 1:
        return [value[0]]
    else:
        print('Unable to handle object')
        sys.exit(1)

def _query_range(data, name, default):
    try:
        value = data[name]
        return _parse_range(value)
    except KeyError:
        return default

def run_experiment(filename):
    try:
        handle = open(filename)
    except IOError,e:
        print(e)
        sys.exit(1)

    data = yaml.load(handle)
    handle.close()

    dimensions = data['dimensions']
    outfile    = data['outfile']
    binary     = data['binary']
    parameters = data['parameters']

    elements_per_thread = _query_range(parameters, 'elements_per_thread', [1])
    problem_size        = _query_range(parameters, 'problem_size', [128])
    time_steps          = _query_range(parameters, 'time_steps', [64])
    time_tile_size      = _query_range(parameters, 'time_tile_size', [1])

    block_size_x = _query_range(parameters, 'block_size_x', [16])
    if dimensions > 1:
        block_size_y = _query_range(parameters, 'block_size_y', [16])
        if dimensions > 2:
            block_size_z = _query_range(parameters, 'block_size_z', [8])
        else:
            block_size_z = [1]
    else:
        block_size_y = [1]
        block_size_z = [1]

    try:
        phase_limit = data['phase_limit']
    except:
        phase_limit = 0

    try:
        counters = data['counters'].split(',')
    except:
        counters = []

    output = open(outfile, 'w')

    num_runs = len(problem_size) * len(time_steps) * len(elements_per_thread) \
             * len(time_tile_size) * len(block_size_x) * len(block_size_y) \
             * len(block_size_z)

    print('Number of Runs: %d' % num_runs)

    curr = 0

    total_start = time.time()

    # Run through each permutation
    for ps in problem_size:
        for ts in time_steps:
            for elems in elements_per_thread:
                for tt in time_tile_size:
                    for bsx in block_size_x:
                        for bsy in block_size_y:
                            for bsz in block_size_z:

                                # Before each run, blow away the nv cache
                                os.system('rm -rf ~/.nv/')

                                curr = curr + 1
                                print('Running %d of %d' % (curr, num_runs))

                                args = [binary,
                                        '-n',
                                        '%d' % ps,
                                        '-t',
                                        '%d' % ts,
                                        '-e',
                                        '%d' % elems,
                                        '-s',
                                        '%d' % tt,
                                        '-x',
                                        '%d' % bsx,
                                        '-p',
                                        '%d' % phase_limit]

                                if dimensions > 1:
                                    args.append('-y')
                                    args.append('%d' % bsy)
                                    if dimensions > 2:
                                        args.append('-z')
                                        args.append('%d' % bsz)

                                args = ' '.join(args)

                                proc = subprocess.Popen(args,
                                                        shell=True,
                                                        stdout=subprocess.PIPE,
                                                        stderr=subprocess.PIPE)

                                # Keep a watchdog on the process
                                start_time = time.time()

                                while proc.poll() == None:
                                    time.sleep(0.1)
                                    now = time.time()
                                    elapsed = now - start_time
                                    if elapsed > 15.0:
                                        print('Watchdog timer expired!')
                                        proc.terminate()
                                        proc.wait()
                                        break

                                end_time = time.time()

                                if proc.returncode != 0:
                                    print('- FAILURE:')
                                    print(proc.stdout.read())
                                    print(proc.stderr.read())
                                else:
                                    for line in proc.stdout.readlines():
                                        output.write('%d#%s' % (curr, line))
                                    output.flush()

                                elapsed = end_time - start_time
                                total   = time.time() - total_start

                                for cnt in counters:
                                    if proc.returncode == 0:
                                        with open('/tmp/experiment-profiler.conf', 'w') as conf:
                                            conf.write(cnt)
                                    prof_args = 'COMPUTE_PROFILE=1 COMPUTE_PROFILE_CONFIG=/tmp/experiment-profiler.conf COMPUTE_PROFILE_LOG=/tmp/prof.log COMPUTE_PROFILE_CSV=1 %s' % args
                                    
                                    proc = subprocess.Popen(prof_args,
                                                            shell=True,
                                                            stdout=subprocess.PIPE,
                                                            stderr=subprocess.PIPE)

                                    # Keep a watchdog on the process
                                    start_time = time.time()

                                    while proc.poll() == None:
                                        time.sleep(0.1)
                                        now = time.time()
                                        elapsed = now - start_time
                                        if elapsed > 15.0:
                                            print('Watchdog timer expired!')
                                            proc.terminate()
                                            proc.wait()
                                            break

                                    end_time = time.time()

                                    if proc.returncode != 0:
                                        print('- FAILURE:')
                                        print(proc.stdout.read())
                                        print(proc.stderr.read())
                                    else:
                                        all_values = []
                                        with open('/tmp/prof.log') as log:
                                            for line in log.readlines():
                                                line = line.strip()
                                                if line.startswith('kernel_func'):
                                                    value = line.split(',')[-1]
                                                    all_values.append(int(value))
                                            #for line in proc.stdout.readlines():
                                            #    output.write('%d#%s' % (curr, line))
                                            value_avg = float(sum(all_values)) / float(len(all_values))
                                            output.write('%d#%s: %f\n' % (curr, cnt.strip(), value_avg))
                                            output.flush()

                                    elapsed = end_time - start_time
                                    total   = time.time() - total_start


                                seconds_per_run = total / float(curr)
                                remaining_runs  = float(num_runs) - float(curr)
                                remaining_secs  = seconds_per_run * \
                                                  remaining_runs

                                remaining_secs = int(remaining_secs)
                                remaining_mins = remaining_secs / 60
                                remaining_secs = remaining_secs % 60
                                remaining_hrs  = remaining_mins / 60
                                remaining_mins = remaining_mins % 60;

                                print('Elapsed: %f  Total: %f  Remaining: %d:%d:%d' % \
                                          (elapsed, total, remaining_hrs,
                                           remaining_mins, remaining_secs))

    output.close()

