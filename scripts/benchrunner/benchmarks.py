 
import subprocess
import sys
import time


def run_benchmark(path, prob_size, time_steps, time_tile_size, elems_per_thread, **kwargs):
    # Build command-line
    args = '%s -n %d -t %d -s %d -e %d' % (path, prob_size, time_steps, time_tile_size, elems_per_thread)

    if 'bsx' in kwargs:
        args = args + ' -x %d' % kwargs['bsx']
    if 'bsy' in kwargs:
        args = args + ' -y %d' % kwargs['bsy']
    if 'bsz' in kwargs:
        args = args + ' -z %d' % kwargs['bsz']
    if 'phase' in kwargs:
        args = args + ' -p %d' % kwargs['phase']

    start = time.time()
    proc = subprocess.Popen(args.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = proc.communicate()
    proc.wait()
    end = time.time()
    
    return (proc.returncode, stdout, stderr, end-start)

