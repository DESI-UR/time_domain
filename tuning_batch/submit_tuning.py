#!/usr/bin/env python
import numpy as np
import os
import sys
from datetime import datetime
from subprocess import call
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser(description='3LabelCNN Batch Tuning submitter.')
parser.add_argument('-x', '--execute', dest='execute', action='store_true',
                    default=False, help='Submit jobs to slurm.')
parser.add_argument('--nologs', action='store_true', default=False,\
  help="If not provided, tensorboard log files are saved")
parser.add_argument('--noweights', action='store_true', default=False,\
  help="If not provided, weight hdf5 are saved")
parser.add_argument('--preprocess_data', action='store_true', default=False,\
  help="If provided, data is preprocessed and saved")
args = parser.parse_args()

basedir = '/scratch/dgandhi/desi/time-domain/tuning_batch_v2'
niter = 100
batch_ct = datetime.now().strftime("%m-%d_%H:%M:%S")
batch_basedir_scripts = "/".join([basedir,'cnn/categorical/batch({})'.format(batch_ct), 'scripts'])
batch_basedir_logs = "/".join([basedir,'cnn/categorical/batch({})'.format(batch_ct), 'logs'])
os.makedirs(batch_basedir_scripts, exist_ok=True)
os.makedirs(batch_basedir_logs, exist_ok=True)

for i in range(niter):
	lr = 10**(np.random.rand()*(-4)-2)
	reg = 10**(np.random.rand()*(-4)-1)
	dropout = np.random.rand()
	epochs = np.random.choice([75, 100, 125], 1)[0]
	bsize = np.random.choice([16,32,64,128], 1)[0]
	
	ct = datetime.now().strftime("%m-%d_%H:%M:%S_%f")
	print(batch_ct, ct, i, lr, reg, dropout, epochs, bsize)

	slurmfile = '/'.join([batch_basedir_scripts, 'tune_iter({})_run({}).sh'.format(i, ct)])
	logfile = '/'.join([batch_basedir_logs, 'tune_iter({})_run({}).log'.format(i, ct)])
	
	params_str = " --batch_time {} --run_time {} --lr {} --reg {} --dropout {} --epochs {} --bsize {} --upper_iter {}".format(batch_ct, ct, lr, reg, dropout, epochs, bsize, i)
	if args.nologs:
		params_str += " --nologs"
	if args.noweights:
		params_str += " --noweights"
	if args.preprocess_data:
		params_str += " --preprocess_data"


	# Write the sbatch script.
	sbatch_script = [
		'#!/bin/bash',
		'#SBATCH --partition=standard --time=20:00:00 --mem=24G --output={}'.format(logfile),
		'date',
		'hostname',
		'source /scratch/sbenzvi_lab/desi/setup_desi_software.sh',
		'cd /scratch/dgandhi/desi/time-domain/tuning_batch_v2',
		'python 3LabelCNNTuning.py' + params_str,
		'date'
	]
	with open(slurmfile, 'w') as sf:
		sf.write('\n'.join(sbatch_script))

	call(['chmod', 'ug+x', slurmfile])

	# Submit jobs to slurm if execute flag is enabled.
	if args.execute:
		call(['sbatch', slurmfile])
