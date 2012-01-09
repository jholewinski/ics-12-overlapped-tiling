#!/usr/bin/env python

import sys

runs = {}

for line in sys.stdin.readlines():
    line = line.strip()
    (run, data) = line.split('#')
    run = int(run)
    data = data.strip()
    (key, value) = data.split(':')
    key = key.strip()
    value = value.strip()
    
    if run not in runs:
        runs[run] = {}

    runs[run][key] = value

    
columns = [
    'Arithmetic Intensity',
    'Global Loads/Block',
    'Global Stores/Block',
    'Shared Loads/Block',
    'Shared Stores/Block',
    'Useful Ratio',
    'Num Blocks (Shared)',
    'Max Blocks',
    'Actual GFlop/s']

for col in columns:
    sys.stdout.write('%s,' % col)
sys.stdout.write('\n')

for (run, values) in runs.iteritems():
    sys.stdout.write('%s' % values[columns[0]])
    for col in columns[1:]:
        sys.stdout.write(',%s' % values[col])
    sys.stdout.write('\n')

sys.stdout.flush()

    
            


