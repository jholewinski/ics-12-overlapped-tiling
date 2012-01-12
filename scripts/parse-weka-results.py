#!/usr/bin/env python

import sys

maximum = 0.0
selected = 0.0

for line in sys.stdin.readlines()[5:]:
    line = line.strip()
    if len(line) == 0:
        continue

    (inst, actual, predicted, error) = line.split()

    predicted = float(predicted)
    if predicted > maximum:
        maximum = predicted
        selected = float(actual)

sys.stdout.write('Maximum Prediction: %f\n' % maximum)
sys.stdout.write('Selected Actual:    %f\n' % selected)
