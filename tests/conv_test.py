from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, Flatten, Dense, LSTM

import random as r
from tqdm import trange

import numpy as np

def datum(size):
    generator = r.choice([
        r.random,
        lambda: r.gauss(0.0, 1.0),
        lambda: r.expovariate(1.0)
    ])

    item = np.asarray([generator() for _ in range(size)])
    sequence = item.reshape((1, size, 1))

    out = np.asarray([
        np.sum(item),
        np.average(item),
        np.std(item),
        np.max(item),
        np.min(item),
        size,
    ]).reshape((1,6))

    return sequence, out

EXAMPLES = 15000

model = Sequential()
model.add(Conv1D(8, 5, input_shape=(None, 1)))
model.add(Conv1D(16, 5, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(6))

# MUCH slower
# model = Sequential()
# model.add(LSTM(16, input_dim=1))
# model.add(Dense(5))

model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.summary()

for e in trange(EXAMPLES):
    sequence, label = datum(500000 if r.random() < 0.001 else r.randint(50, 100))
    model.train_on_batch(sequence, label)

for _ in range(5):
    sequence, label = datum(r.randint(50, 100))
    print(label)
    print(model.predict(sequence))
    print()

for _ in range(5):
    sequence, label = datum(r.randint(50, 500000))
    print(label)
    print(model.predict(sequence))
    print()
