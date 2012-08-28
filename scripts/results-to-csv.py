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

    line = ''
    error = False
    for col in columns:
        try:
            val = values[col]
        except:
            sys.stderr.write('Warning: missing %s for run %d\n' % (col, run))
            val = '0'
            error = True
        line = line + ('%s,' % val)

    if not error:
        sys.stdout.write('%s\n' % line)
    else:
        sys.stderr.write('Skipping run %d\n' % run)

sys.stdout.flush()

