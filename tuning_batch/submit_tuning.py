#!/usr/bin/env python
import numpy as np
import os
import sys
from astropy.io import fits
from datetime import datetime
from subprocess import call
from glob import glob
from argparse import ArgumentParser

#Arguments
parser = ArgumentParser(description='3LabelCNN Batch Tuning submitter.')
parser.add_argument('-x', '--execute', dest='execute', action='store_true',
  default=False, help='Submit jobs to slurm.')
parser.add_argument('--noevents', action='store_true', default=False,\
  help="If not provided, tensorboard event files are saved")
parser.add_argument('--noweights', action='store_true', default=False,\
  help="If not provided, weight hdf5 are saved")
parser.add_argument('--num_iters', type=int, default=100,\
  help="Number of iterations to run")
args = parser.parse_args()

#Main Directory of everything. Change if workspace changes
basedir = '/scratch/dgandhi/desi/time_domain/tuning_batch'

#Check if all fits files are preprocessed
isfile = os.path.isfile
exists_ = isfile("/".join([basedir,'x_train.fits'])) and isfile("/".join([basedir,'x_test.fits']))
exists_ = exists_ and isfile("/".join([basedir,'y_train.fits'])) and isfile("/".join([basedir,'y_test.fits']))

if not exists_:
	print("run preprocessing.py in order to get the appropriate data files before running tuning")
	sys.exit()	

#Set up directories
#Batch name is the time we start the batch
batch_ct = datetime.now().strftime("%m-%d_%H:%M:%S")
batch_basedir_scripts = "/".join([basedir,'cnn/categorical/batch({})'.format(batch_ct), 'scripts'])
batch_basedir_logs = "/".join([basedir,'cnn/categorical/batch({})'.format(batch_ct), 'logs'])
os.makedirs(batch_basedir_scripts, exist_ok=True)
os.makedirs(batch_basedir_logs, exist_ok=True)


#Start iterations
niter = args.num_iters
for i in range(niter):
	#Hyper-parameters to search for
	#TODO: Start limiting search space?
	lr = 10**(np.random.rand()*(-4)-2)
	reg = 10**(np.random.rand()*(-4)-1)
	dropout = np.random.rand()
	epochs = np.random.choice([75, 100, 125], 1)[0]
	bsize = np.random.choice([16,32,64,128], 1)[0]

	# Model name is the time we start the training	
	ct = datetime.now().strftime("%m-%d_%H:%M:%S_%f")

	# Slurm and Log file names, based on iteration number and runtime
	slurmfile = '/'.join([batch_basedir_scripts, 'tune_iter({})_run({}).sh'.format(i, ct)])
	logfile = '/'.join([batch_basedir_logs, 'tune_iter({})_run({}).log'.format(i, ct)])
	
	# Set up parameters for function call	
	params_str = " --batch_time {} --run_time {} --lr {} --reg {} --dropout {} --epochs {} --bsize {} --upper_iter {}".format(batch_ct, ct, lr, reg, dropout, epochs, bsize, i)
	if args.noevents:
		params_str += " --noevents"
	if args.noweights:
		params_str += " --noweights"
	
	# Write the sbatch script.
	sbatch_script = [
		'#!/bin/bash',
		'#SBATCH --partition=standard --time=20:00:00 --mem=24G --output={}'.format(logfile),
		'date',
		'hostname',
		'source /scratch/sbenzvi_lab/desi/setup_desi_software.sh',
		'cd /scratch/dgandhi/desi/time_domain/tuning_batch',
		'python 3LabelCNNTuning.py' + params_str,
		'date'
	]
	with open(slurmfile, 'w') as sf:
		sf.write('\n'.join(sbatch_script))

	# Enable permissions to run
	call(['chmod', 'ug+x', slurmfile])

	# Submit jobs to slurm if execute flag is enabled.
	if args.execute:
		call(['sbatch', slurmfile])
