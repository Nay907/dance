import numpy as np
import os

from keras.layers import LSTM, Dense, Input, LeakyReLU 
from keras.models import Model 
from keras.losses import CategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

data_size=8
init = False

l = []
lD = {}
c = 0
for npy in os.listdir():
	if npy.split('.')[-1] == "npy" and not(npy.split('.')[0]=="labels"):
		print(npy.split('.')[0])
		l.append(str(npy.split('.')[0]))
		lD[str(npy.split('.')[0])] = c
		c = c+1
		if not(init):
			init = True
			arr = np.load(npy)
			print("="*50)
			print(arr.shape)
			labels = np.array([str(npy.split('.')[0])]*data_size)

		else:
			print("="*75)
			print(npy)
			ds = np.load(npy)
			print(ds.shape)
			arr = np.concatenate((arr, ds))
			labels = np.concatenate((labels, np.array([str(npy.split('.')[0])]*data_size)))

print("="*100)

print(labels)

for m, i in enumerate(labels):
	labels[m] = lD[i]

print(labels)
#print(labels)
labels = to_categorical(labels)
#print(labels)
print(l, lD)
print("="*100)
#print(labels)
print("="*50)
print("shape of x is : ", arr.shape, labels.shape)

data = []
for i in range(data_size*len(l)):
	data.append([labels[i], arr[i]])


data = np.array(data)

print("= "*50)
np.random.shuffle(data)

X = []
for i in data[:, 1]: 
	X.append(i)
Y = []
for i in data[:, 0]:
	Y.append(i)

X = np.array(X)
Y = np.array(Y)

print(X.shape, Y.shape)

#Y = data[:, 1]

x = Input(shape=(24,66))

y = LSTM(128, return_sequences=True)(x)
y = LSTM(64, return_sequences=False)(x)

y = Dense(32, activation="tanh")(y)

op = Dense(len(l), activation="softmax")(y)

model = Model(x, op)

model.summary()

model.compile(optimizer='rmsprop', loss=CategoricalCrossentropy(), metrics=['acc'])
model.fit(X, Y, epochs=100)
model.save('model.h5')

np.save("labels.npy", np.array(l))
