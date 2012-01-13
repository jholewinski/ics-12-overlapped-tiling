#!/usr/bin/env python

import copy
import optparse
import subprocess
import sys


benchmarks = [
    'jacobi-1d',
    'jacobi-2d',
    'jacobi-3d',
    'gradient-2d',
    'fdtd-2d',
    'poisson-2d',
    'rician-2d'
]

block_sizes = [
    [16, 8],
    [32, 8],
    [48, 8],
    [64, 8],
    [80, 8],

#    [96, 4],
#    [32, 12],
#    [80, 4],
#    [16, 4]

#    [64, 4]
]


parser = optparse.OptionParser()
parser.add_option('-j', '--jar', dest='jar',
                  action='store', type='string',
                  help='WEKA Class Path')
parser.add_option('-c', '--classifier', dest='classifier',
                  action='store', type='string',
                  default='weka.classifiers.lazy.IBk',
                  help='WEKA Classifier')
parser.add_option('-d', '--device', dest='device',
                  action='store', type='string',
                  help='Device name')
parser.add_option('-a', '--train-all', dest='train_all',
                  action='store_true', default=False,
                  help='Train on all block sizes')
parser.add_option('-t', '--test-all', dest='test_all',
                  action='store_true', default=False,
                  help='Test on all block sizes')




(options, args) = parser.parse_args()

weka_log = open('weka.log', 'w')

for left_out in benchmarks:

    print('Testing on %s' % left_out)

    remaining = copy.deepcopy(benchmarks)
    remaining.remove(left_out)

    training_file = open('training.csv', 'w')
    first = True

    # Build training data
    for trainer in remaining:
        if options.train_all:
            data = open('%s-scaling.%s.csv' % (trainer, options.device))
            args = ['../../scripts/extract-features.py']
            if first:
                args.append('-p')
                first=False
            subprocess.call(args, stdout=training_file, stdin=data)
            data.close()
        else:
            for bs in block_sizes:
                data = open('%s-scaling.%s.csv' % (trainer, options.device))
                args = ['../../scripts/extract-features.py']
                args.append('-x')
                args.append(str(bs[0]))
                args.append('-y')
                args.append(str(bs[1]))
                if first:
                    args.append('-p')
                    first=False
                subprocess.call(args, stdout=training_file, stdin=data)
                data.close()

    training_file.close()

    # Build the model
    args = ['java', '-cp', options.jar, options.classifier, '-t',
            'training.csv', '-d', 'testing.model']
    subprocess.call(args, stdout=weka_log, stderr=weka_log)

    # Build the test data
    test_file = open('test.csv', 'w')

    first = True
    if options.test_all:
        args = ['../../scripts/extract-features.py']
        if first:
            args.append('-p')
            first = False
        data = open('%s-scaling.%s.csv' % (left_out, options.device))
        subprocess.call(args, stdout=test_file, stdin=data)
        data.close()
    else:
        for bs in block_sizes:
            args = ['../../scripts/extract-features.py']
            args.append('-x')
            args.append(str(bs[0]))
            args.append('-y')
            args.append(str(bs[1]))
            if first:
                args.append('-p')
                first = False
            data = open('%s-scaling.%s.csv' % (left_out, options.device))
            subprocess.call(args, stdout=test_file, stdin=data)
            data.close()

    test_file.close()

    # Test the model
    args = ['java', '-cp', options.jar, options.classifier, '-T',
            'test.csv', '-l', 'testing.model', '-p', '0']
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    args = ['../../scripts/parse-weka-results.py']
    subprocess.call(args, stdin=proc.stdout)

   
weka_log.close()
