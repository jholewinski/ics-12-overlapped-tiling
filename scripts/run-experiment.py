#!/usr/bin/env python

import utils
import sys

# Get experiment file from command-line argument
for file in sys.argv[1:]:
    print('=== Running %s' % file)
    utils.run_experiment(file)

