import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels") and i != "emotion.npy":
        if not is_init:
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            input_shape = X.shape[1]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            new_data = np.load(i)
            if new_data.shape[0] != size:
                raise ValueError(f"Data size mismatch: {i}")
            X = np.concatenate((X, new_data))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

ip = Input(shape=(input_shape,))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
model.fit(X_new, y_new, epochs=50)  # Training with shuffled data

model.save("model.h5")
np.save("labels.npy", np.array(label))
