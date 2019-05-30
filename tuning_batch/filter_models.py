#bin/src/env python
import sys
import os
import glob
import json
import argparse
#Arguments
parser = argparse.ArgumentParser(description='file_filter')
parser.add_argument('--batch_time', type=str, required=True, \
  help="Batch time, the name of the directory we're searching for potentials CNNs")
parser.add_argument('--min_dropout', type=float, default=0.4,\
  help ="Minimum dropout constant")
parser.add_argument('--min_acc', type=float, default=0.95, \
  help="Minimum accuracy")
parser.add_argument('--output_file', type=str, default="filter_out.txt",\
  help="Save filters")
args = parser.parse_args()
#Path file, change if workspace directory changes
path = '/scratch/dgandhi/desi/time_domain/tuning_batch/cnn/categorical/batch({})'.format(args.batch_time)
json_files = glob.glob("/".join([path,'*/hist.json']))
output_text = open("/".join([path, args.output_file]), 'w')

for jfile in json_files:
	with open(jfile, 'r') as f:
		jdata = json.load(f)
		jmax_acc = max(jdata['acc'])
		jdropout = jdata['dropout']
		if jdropout > args.min_dropout  and jmax_acc > args.min_acc:
			runtime_file =  ((jfile.rsplit('/', 2))[-2])
			print(jmax_acc, jdropout, runtime_file)
			output_text.write("\t".join([str(jmax_acc), str(jdropout),runtime_file])+"\n")
			output_text.flush()
	f.close()
output_text.close()
