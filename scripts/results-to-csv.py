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

    
first = True
columns = []

for (run, values) in runs.iteritems():
    if first:
        for (key, value) in values.iteritems():
            columns.append(key)
        first = False
        for col in columns:
            sys.stdout.write('%s,' % col)
        sys.stdout.write('\n')

    for col in columns:
        sys.stdout.write('%s,' % values[col])
    sys.stdout.write('\n')

sys.stdout.flush()

    
            


