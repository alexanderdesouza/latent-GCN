from __future__ import print_function
import random

random.seed(0)

PROP_TEST = 0.2

with open('rita.cites', 'r') as file:

    lines = [line.strip() for line in file]

    random.shuffle(lines)

    piv = int(len(lines) * PROP_TEST)
    test = lines[:piv]
    train = lines[piv:]

    with open('rita.test.cites', 'w') as out:
        for line in test:
            print(line, file=out)
    with open('rita.train.cites', 'w') as out:
        for line in test:
            print(line, file=out)