#!/usr/bin/env python
import numpy as np
import os
import sys
from datetime import datetime
from subprocess import call
from argparse import ArgumentParser

#Arguments
parser = ArgumentParser(description='3LabelCNN Batch Tuning submitter.')
parser.add_argument('-x', '--execute', dest='execute', action='store_true',
  default=False, help='Submit jobs to slurm.')
args = parser.parse_args()

#Main Directory of everything. Change if workspace changes
basedir = '/scratch/dgandhi/desi/time-domain-bkup/tuning_batch_v2'

# Slurm and Log file names, based on iteration number and runtime
slurmfile = '/'.join([basedir, 'preprocess.sh'])
logfile = '/'.join([basedir, 'preprocess.log'])
	
# Write the sbatch script.
sbatch_script = [
	'#!/bin/bash',
	'#SBATCH --partition=standard --time=20:00:00 --mem=30G --output={}'.format(logfile),
	'date',
	'hostname',
	'source /scratch/sbenzvi_lab/desi/setup_desi_software.sh',
	'cd /scratch/dgandhi/desi/time-domain-bkup/tuning_batch_v2',
	'python preprocess_data.py',
	'date'
]
with open(slurmfile, 'w') as sf:
	sf.write('\n'.join(sbatch_script))

# Enable permissions to run
call(['chmod', 'ug+x', slurmfile])

# Submit jobs to slurm if execute flag is enabled.
if args.execute:
	call(['sbatch', slurmfile])
