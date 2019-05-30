#!/usr/bin/env python
from numpy.random import seed
seed(1)
import json
import numpy as np
import argparse
from datetime import datetime
import os
import glob
from sklearn.model_selection import train_test_split
from astropy.io import fits
from matplotlib import pyplot as plt
from keras import regularizers, callbacks
from keras.utils.np_utils import to_categorical
from keras.layers import (Input, Dense, Activation, ZeroPadding1D, 
BatchNormalization, Flatten, Reshape, Conv1D, MaxPooling1D, Dropout,Add, LSTM,Embedding)
from keras.initializers import glorot_normal, glorot_uniform
from keras.optimizers import Adam
from keras.models import Model, load_model

# Function that creates and returns network
def network(input_shape, learning_rate=0.0001, reg=0.032, dropout=0.7436, seed=None):
    """ 
    Args:
    input_shape -- shape of the input spectra
    regularization_strength -- regularization factor
    dropout -- dropout rate
    seed -- seed of initializer
    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape, name='Input_Spec')

    f = [8, 16, 32, 64, 128]
    k = [5, 5, 5, 5, 5]

    X = X_input                       
    for i in range(3):
        X = Conv1D(filters=f[i], kernel_size=k[i], strides=1, padding='same',
                 kernel_regularizer=regularizers.l2(reg),
                 bias_initializer='zeros',
                 kernel_initializer='glorot_normal')(X)
        X = BatchNormalization(axis=2)(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(pool_size= 2)(X)
        
    # FLATTEN -> FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(256*2, kernel_regularizer=regularizers.l2(reg),
                  activation='relu')(X)
    X = Dropout(rate=dropout)(X)
    X = Dense(3, kernel_regularizer=regularizers.l2(reg),
              activation='softmax', name='Output_Classes')(X)

    model = Model(inputs=X_input, outputs=X, name='SNnet')
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data():
	#Permute and set up the training data
	h = fits.open('x_train.fits')
	x_train_ = h[0].data
	h.close()
	h = fits.open('y_train.fits')
	y_train_ = h[0].data
	h.close()
	h = fits.open('x_test.fits')
	x_test_ = h[0].data
	h.close()
	h = fits.open('y_test.fits')
	y_test_ = h[0].data
	h.close()
	
	permute = np.random.permutation(y_train_.shape[0])
	permute_test = np.random.permutation(y_test_.shape[0])
	x_train = x_train_[permute]
	y_train_ = y_train_[permute]
	y_train = to_categorical(y_train_)	
	x_test = x_test_[permute_test]
	y_test_ = y_test_[permute_test]
	y_test = to_categorical(y_test_)	
	return x_train, x_test, y_train, y_test

def main():
	parser = argparse.ArgumentParser(description='DESI SN-Net script')
	parser.add_argument('--noevents', action='store_true', default=False,\
	  help="If not provided, tensorboard event files are saved to directory specified by the log_dir_ variable")
	parser.add_argument('--noweights', action='store_true', default=False,\
	  help="If not provided, weight hdf5 are saved to directory specified by the basedir variable")
	parser.add_argument('--batch_time', type=str, default=datetime.now().strftime("%m-%d_%H:%M:%S"),\
	  help="Time/Name of the head directory")
	parser.add_argument('--run_time', type=str, default=datetime.now().strftime("%m-%d_%H:%M:%S_%f"),\
	  help="Time/Name of the current model")
	parser.add_argument('--upper_iter', type=int, default=-1,\
	  help="Iteration number at upper level")
	parser.add_argument('--lr', type=float, default=0.0001,\
	  help="Learning rate")
	parser.add_argument('--reg', type=float, default=0.032,\
	  help="Regularization constant")
	parser.add_argument('--dropout', type=float, default=0.7436,\
	  help ="Dropout constant")
	parser.add_argument('--epochs', type=int, default=100, \
	  help="Number of epochs through the full training data to perform.")
	parser.add_argument('--bsize', type=int, default=64,\
	  help="The batch size for the training data")
	args = parser.parse_args()

    	#load the data
	print("Loading data....")
	x_train, x_test, y_train, y_test = load_data()

	#create directory for specific model
	basedir = '/scratch/dgandhi/desi/time_domain/tuning_batch/cnn/categorical/batch({})/iter({})_run({})'.format(args.batch_time,args.upper_iter,args.run_time)
	os.makedirs(basedir, exist_ok=True)
	callbacks_ = []

	if not args.noweights:
		path = "/".join([basedir, 'weights'])
		os.makedirs(path, exist_ok=True)
		path = path+'/weights.Ep{epoch:02d}-ValAcc{val_acc:.2f}.hdf5'
		checkpoint = callbacks.ModelCheckpoint(path, monitor='val_acc', verbose=1,\
						   save_best_only=True, mode='max',)
		callbacks_.append(checkpoint)
		print("Callbacks for weights set")
	
	if not args.noevents:
		#K.clear_session()
		log_dir_ = "/".join([basedir, 'tensorboard'])
		os.makedirs(log_dir_, exist_ok=True)
		tb = callbacks.TensorBoard(log_dir=log_dir_, batch_size=args.bsize, \
		 write_graph=True, write_images=True, write_grads=True,)
		callbacks_.append(tb)
		print("Callbacks for tensorboard logs set")


	print("Start Model.fit")
	model = network((400,1), learning_rate=args.lr, reg=args.reg, dropout=args.dropout)
	history = model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test), batch_size=args.bsize, epochs=args.epochs,shuffle=True, callbacks=callbacks_, verbose=2)
	
	params = {'batch_time': args.batch_time, 'run_time':args.run_time, 'upper_iter': args.upper_iter, 'reg':args.reg, 'dropout':args.dropout,
	'epochs':args.epochs, 'batch_size':args.bsize}
	params.update(history.history)
	#Print to file
	with open("/".join([basedir, 'hist.json']), 'w') as f:
		json.dump(params, f)


if __name__== '__main__':
	main()
