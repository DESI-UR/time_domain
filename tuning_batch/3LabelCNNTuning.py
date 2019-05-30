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


#Preprocess our data
def preprocess_data():
	print("Loading Host Galaxies")
	#Host Galaxies
	files_host = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/hosts/*/*coadd.fits'))
	flux_host = []
	for f in files_host:
		h = fits.open(f)
		fl = h[1].data
		flux_host.append(fl)
		h.close()
	fluxes_hosts = np.concatenate(flux_host)
	nonzero_hosts = fluxes_hosts.sum(axis=1)!=0
	fluxes_hosts = fluxes_hosts[nonzero_hosts]
	subspec_hosts = np.median(fluxes_hosts[:,:6000].reshape(-1,400,15),2)
	maxflux = fluxes_hosts.max(axis=-1).reshape(-1,1)
	minflux = fluxes_hosts.min(axis=-1).reshape(-1,1)
	standarized_hosts = (subspec_hosts - minflux)/(maxflux-minflux)
	del minflux, maxflux, flux_host, subspec_hosts, fluxes_hosts
	print("Loading Type 1As")

	#Type 1A
	files = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/sne_ia/*/*coadd.fits'))
	flux = []
	for f in files:
		h = fits.open(f)
		f = h[1].data
		zeros = np.zeros(400)
		flux.append(f)
		h.close()
	fluxes = np.concatenate(flux)
	nonzero = fluxes.sum(axis=1)!=0
	fluxes = fluxes[nonzero]
	subspec = np.median(fluxes[:,:6000].reshape(-1,400,15),2)
	maxflux = fluxes.max(axis=-1).reshape(-1,1)
	minflux = fluxes.min(axis=-1).reshape(-1,1)
	standarized = (subspec - minflux)/(maxflux-minflux)
	del minflux, maxflux, flux, subspec, fluxes
	print("Loading Type 2Ps")

	#Type 2P
	files_iip = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/sne_iip/*/*coadd.fits'))
	flux_iip = []
	for f in files_iip:
		h = fits.open(f)
		f = h[1].data
		flux_iip.append(f)
		h.close()
	fluxes_iip = np.concatenate(flux_iip)
	nonzero_iip = fluxes_iip.sum(axis=1)!=0
	fluxes_iip = fluxes_iip[nonzero_iip]
	subspec_iip = np.median(fluxes_iip[:,:6000].reshape(-1,400,15),2)
	maxflux = fluxes_iip.max(axis=-1).reshape(-1,1)
	minflux = fluxes_iip.min(axis=-1).reshape(-1,1)
	standarized_iip = (subspec_iip - minflux)/(maxflux-minflux)
	del minflux, maxflux, flux_iip, subspec_iip, fluxes_iip	
	print("Loading Truth Tables")
	
	#Loading Truth Tables
	#sne_iip
	info_files = ['/'.join(f.split('/')[:-1])+ '/{}truth.fits'.format(f.split('/')[-1][:-10]) for f in files_iip]
	#info_files = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/sne_iip/*/*truth.fits'))
	rfr_ = []
	truez_ = []
	rmags_ =[]
	epochs_=[]
	for f in info_files:
		h = fits.open(f)
		r= h[3].data['SNE_FLUXRATIO']
		rfr_.append(r)
		z = h[3].data['TRUEZ']
		truez_.append(z)
		m = 22.5 - 2.5*np.log10(h[3].data['FLUX_R'])
		rmags_.append(m)
		e=h[3].data['SNE_EPOCH']
		epochs_.append(e)
		h.close()
	rfr_iip = np.concatenate(rfr_)[nonzero_iip]
	truez_iip = np.concatenate(truez_)[nonzero_iip]
	rmags_iip = np.concatenate(rmags_)[nonzero_iip]
	epochs_iip = np.concatenate(epochs_).astype(int)[nonzero_iip]
	del rfr_, truez_, rmags_, epochs_

	#sne_ia
	info_files = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/sne_ia/*/*truth.fits'))
	rfr_ = []
	truez_ = []
	rmags_ =[]
	epochs_=[]
	for f in info_files:
		h = fits.open(f)
		r= h[3].data['SNE_FLUXRATIO']
		rfr_.append(r)
		z = h[3].data['TRUEZ']
		truez_.append(z)
		m = 22.5 - 2.5*np.log10(h[3].data['FLUX_R'])
		rmags_.append(m)
		e=h[3].data['SNE_EPOCH']
		epochs_.append(e)
		h.close()
	rfr_ia = np.concatenate(rfr_)[nonzero]
	truez_ia = np.concatenate(truez_)[nonzero]
	rmags_ia = np.concatenate(rmags_)[nonzero]
	epochs_ia = np.concatenate(epochs_).astype(int)[nonzero]
	del rfr_, truez_, rmags_, epochs_

	#Hosts
	info_files = info_files = ['/'.join(f.split('/')[:-1])+ '/{}truth.fits'.format(f.split('/')[-1][:-10]) for f in files_host]
	rfr_ = []
	truez_ = []
	rmags_ =[]
	epochs_=[]
	for f in info_files:
		h = fits.open(f)
		z = h[3].data['TRUEZ']
		truez_.append(z)
		m = 22.5 - 2.5*np.log10(h[3].data['FLUX_R'])
		rmags_.append(m)
		h.close()
	truez_hosts = np.concatenate(truez_)[nonzero_hosts]
	rmags_hosts = np.concatenate(rmags_)[nonzero_hosts]
	del rfr_, truez_, rmags_, epochs_
	
	print("Pre-processing and returning data")
	#Clean up NaN data. This was added to get rid of some warnings that were happening with arithmetic comparisons on NaN
	rfr_ia_clean = np.array([a if ~np.isnan(a) else 0 for a in rfr_ia])
	rfr_iip_clean = np.array([a if ~np.isnan(a) else 0 for a in rfr_iip])

	#Get the data we're training on, and 3 labels for them
	x_data = np.concatenate([standarized_hosts[:10000], standarized[rfr_ia_clean>0.9], standarized_iip[rfr_iip_clean>0.9]]).reshape(-1,400,1)
	y_labels = np.concatenate([np.zeros(10000), np.ones(standarized[rfr_ia_clean>0.9].shape[0]), 1+np.ones(standarized_iip[rfr_iip_clean>0.9].shape[0])])
	
	x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_labels, test_size=0.1, shuffle=True)	

	x_train_hdu = fits.PrimaryHDU(x_data)
	x_train_hdu.writeto('x_train.fits', overwrite=True)

	y_train_hdu = fits.PrimaryHDU(y_labels)
	y_train_hdu.writeto('y_train.fits', overwrite=True)

	x_test_hdu = fits.PrimaryHDU(x_valid)
	x_test_hdu.writeto('x_test.fits', overwrite=True)

	y_test_hdu = fits.PrimaryHDU(y_valid)
	y_test_hdu.writeto('y_test.fits', overwrite=True)
	return;

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
	print(x_train.shape, y_train.shape)
	return x_train, x_test, y_train, y_test

def main():
	parser = argparse.ArgumentParser(description='DESI SN-Net script')
	parser.add_argument('--preprocess_data', action='store_true', default=False, \
	  help="If provided, program will regenerate x_train.fits and y_train.fits from original coadded data")
	parser.add_argument('--nologs', action='store_true', default=False,\
	  help="If not provided, tensorboard log files are saved to directory specified by the log_dir_ variable")
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

	#preprocess data if needed to
	if args.preprocess_data:
		print("preprocessing data...")	
		preprocess_data()

    	#load the data
	print("Loading data....")
	x_train, x_test, y_train, y_test = load_data()

	#create directory for specific model
	basedir = '/scratch/dgandhi/desi/time-domain/tuning_batch_v2/cnn/categorical/batch({})/iter({})_run({})'.format(args.batch_time,args.upper_iter,args.run_time)
	os.makedirs(basedir, exist_ok=True)
	callbacks_ = []
	#output_text is the file of accuracies outputted
	#output_text = open("/".join([basedir, "accs.txt"]), 'w')

	if not args.noweights:
		path = "/".join([basedir, 'weights'])
		os.makedirs(path, exist_ok=True)
		path = path+'/weights.Ep{epoch:02d}-ValAcc{val_acc:.2f}.hdf5'
		checkpoint = callbacks.ModelCheckpoint(path, monitor='val_acc', verbose=1,\
						   save_best_only=True, mode='max',)
		callbacks_.append(checkpoint)
		print("Callbacks for weights set")
	
	if not args.nologs:
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
	#output_text.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"\
#.format(args.batch_time, args.run_time, args.upper_iter, args.lr, args.reg, args.dropout, args.epochs, args.bsize, history.history['acc'], history.history['val_acc']))
	#output_text.flush()
	#output_text.close()



if __name__== '__main__':
	main()
