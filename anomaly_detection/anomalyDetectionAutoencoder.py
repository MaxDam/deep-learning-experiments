import json
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Lambda, Input, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras.losses import mse, binary_crossentropy
from keras import backend as K
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
#%matplotlib inline

batch_size=32
epochs=1
lr=0.001

#autoencoder
input_dim = 4
hidden_dim = 3
encoding_dim = 2
dropout = 0.3
inputLayer = Input(shape=(input_dim,))
encoderLayer = Dense(hidden_dim, activation="relu")(inputLayer)
#encoderLayer = Dropout(dropout)(encoderLayer)
encoderLayer = Dense(encoding_dim, activation="relu")(encoderLayer)
decoderLayer = Dense(hidden_dim, activation='relu')(encoderLayer)
#decoderLayer = Dropout(dropout)(decoderLayer)
decoderLayer = Dense(input_dim, activation='sigmoid')(decoderLayer)
model = Model(inputs=inputLayer, outputs=decoderLayer)
optimizer = Adam(lr=lr)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

'''
history = model.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.3, verbose=1)
score = model.evaluate(X, X, verbose=0)
print(score)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

def generator(data, batch_size):
    counter=0
    while True:
        batch = np.array(data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        counter += 1
        if(len(batch) > 0): counter = 0
        yield batch, batch

path = './data'
for filename in sorted(os.listdir(path)):
    print("File:" + filename)
    df = pd.read_csv(path+"/"+filename, sep='\t')
    data = df.iloc[:, :].values.astype(np.float32)
    #data = shuffle(data)
    data_train, data_test, _, _ = train_test_split(data, data, test_size=0.3, random_state=42)
    history = model.fit_generator(generator(data_train, batch_size), 
        epochs=epochs, verbose=1, shuffle=True, steps_per_epoch=data.shape[0]/batch_size,
        validation_data=generator(data_test, batch_size),validation_steps=data.shape[0]/batch_size)
    print(history.history)