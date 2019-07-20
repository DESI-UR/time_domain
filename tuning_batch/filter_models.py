#bin/src/env python
import sys
import os
import glob
import json
import argparse
#Arguments
parser = argparse.ArgumentParser(description='file_filter')
parser.add_argument('--batch_dir', type=str, required=True, \
  help="The directory we're searching for potentials CNNs")
parser.add_argument('--max_dropout', type=float, default=0.4,\
  help ="Maximum dropout constant")
parser.add_argument('--min_acc', type=float, default=0.95, \
  help="Minimum accuracy")
parser.add_argument('--output_file', type=str, default="filter_out.txt",\
  help="Save filters")
args = parser.parse_args()
#Path file, change if workspace directory changes
#path = '/scratch/dgandhi/desi/time-domain-bkup/tuning_batch_v2/cnn/categorical/batch({})'.format(args.batch_time)
path = args.batch_dir
json_files = glob.glob("/".join([path,'*/hist.json']))
output_text = open("/".join([path, args.output_file]), 'w')

output_text.write("\t".join(["max_val_acc", "dropout", "LR", "Reg", "batchsize", "iter#"]))
output_text.write("\n")
for jfile in json_files:
	with open(jfile, 'r') as f:
		jdata = json.load(f)
		jmax_acc = max(jdata['val_acc'])
		jdropout = jdata['dropout']
		#print(jfile)
		if jdropout < args.max_dropout  and jmax_acc > args.min_acc:
			runtime_file =  ((jfile.rsplit('/', 2))[-2])
			print(runtime_file)
			output_text.write("\t".join([str(jmax_acc), str(jdropout), str(jdata['lr']), str(jdata["reg"]),str(jdata["batch_size"]), str(jdata["upper_iter"]),"\n"]))
			output_text.flush()
	f.close()
output_text.close()
