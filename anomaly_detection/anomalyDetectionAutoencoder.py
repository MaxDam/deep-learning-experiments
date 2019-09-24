import json
import numpy as np
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
import dataLayer as dl
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#%matplotlib inline

#set del seed per la randomizzazione
np.random.seed(1671)

#Autoencoder
class Autoencoder(object):

    def __init__(self):
        self.autoencoderModelFile = 'tmp/ae.json'
        self.autoencoderWeightFile= 'tmp/ae.h5'
        self.verbose=1

        #hyperparameters
        self.use_dropout = True
        self.use_autodimension = True
        self.encoding_dim = 32
        #self.encoding_dim = 16
        self.hidden_dim = self.encoding_dim * 2
        self.dropout = 0.3
        self.batch_size=32
        self.epochs=200
        self.shuffle=False
        self.validation_split=0.3
        self.lr=0.001
        
    def buildAndFit(self, X):
        #log
        print("ADDESTRAMENTO AUTOENCODER")

        #attrubuisce le dimensioni degli strati in automatico in base alla dimensione dell'input
        if self.use_autodimension:
            self.encoding_dim = int(X.shape[1] / 4)
            self.hidden_dim = int(X.shape[1] / 2)

        #crea il modello autoencoder
        in_out_neurons_number = X.shape[1]
        inputLayer = Input(shape=(in_out_neurons_number, ))
        encoderLayer = Dense(self.hidden_dim, activation="tanh")(inputLayer)
        if self.use_dropout:
            encoderLayer = Dropout(self.dropout)(encoderLayer)
        encoderLayer = Dense(self.encoding_dim, activation="tanh")(encoderLayer)
        decoderLayer = Dense(self.hidden_dim, activation='tanh')(encoderLayer)
        if self.use_dropout:
            decoderLayer = Dropout(self.dropout)(decoderLayer)
        decoderLayer = Dense(in_out_neurons_number, activation='sigmoid')(decoderLayer)
        self.model = Model(inputs=inputLayer, outputs=decoderLayer)
        self.encoder = Model(inputLayer, encoderLayer)

        #inizializza la tensorboard per l'autoencoder (http://localhost:6006/)
        tensorboard = TensorBoard(log_dir='./logs/autoencoder', histogram_freq=0, write_graph=True, write_images=True)

        #compila l'autoencoder
        optimizer = Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

        #addestra l'autoencoder
        self.history = self.model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, shuffle=self.shuffle, validation_split=self.validation_split, verbose=self.verbose, callbacks=[tensorboard])
        
        #torna il punteggio del training
        score = self.model.evaluate(X, X, verbose=0)
        return score

    def predict(self, X):
        return self.encoder.predict(X)

    def save(self):
        self.encoder.save(self.autoencoderModelFile)
        self.encoder.save_weights(self.autoencoderWeightFile)

    def load(self):
        self.encoder = load_model(self.autoencoderModelFile)
        self.encoder.load_weights(self.autoencoderWeightFile)

    def plotTrain(self):
        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


ae = ann2.Autoencoder()
ae.verbose=0

scoreAE = ae.buildAndFit(X)
X = ae.predict(X)
scoreANN = fc.buildAndFit(X, Y)
ae.plotTrain()
print('AE test score:', scoreAE[0])
print('AE test accuracy:', scoreAE[1])