#!/usr/bin/env python

import sys

maximum = 0.0
selected = 0.0

results = []

for line in sys.stdin.readlines()[5:]:
    line = line.strip()
    if len(line) == 0:
        continue

    (inst, actual, predicted, error) = line.split()
    results.append([inst, actual, predicted, error])

    predicted = float(predicted)
    if predicted > maximum:
        maximum = predicted
        selected = float(actual)

by_predicted = sorted(results, key=lambda entry: float(entry[2]))
by_predicted.reverse()
by_actual = sorted(results, key=lambda entry: float(entry[1]))
by_actual.reverse()

best_of_actuals = float(by_actual[0][1])
sys.stdout.write('Best of Actuals:    %f\n' % best_of_actuals)
sys.stdout.write('Maximum Prediction: %s\n' % 
                 str([x[2] for x in by_predicted[0:5]]))
sys.stdout.write('Selected Actual:    %s\n' %
                 str([x[1] for x in by_predicted[0:5]]))
sys.stdout.write('Percentages:        %s\n' %
                 str([float(x[1])/best_of_actuals for x in by_predicted[0:5]]))


