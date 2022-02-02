#!/usr/bin/env python3

from argparse import ArgumentParser
from hgcal_predictor import HGCalPredictor
import setGPU

parser = ArgumentParser('Apply a model to a (test) source sample.')
parser.add_argument('inputModel')
parser.add_argument('inputData',
                    help="Data collection file in djcdc format from which to pick files to run inference on. Can also be a text file containing input files.")
parser.add_argument('--dc',
                    help="Data collection to infer the data class from (not used as input, and only needed if input file is a text file)",
                    default=None)
parser.add_argument('output_dir', help="will be created if it doesn't exist.")
parser.add_argument("--unbuffered",
                    help="do not read input in memory buffered mode (for lower memory consumption on fast disks)",
                    default=False, action="store_true")

parser.add_argument("--max_files", help="Limit number of files", default=-1)

args = parser.parse_args()

'''
self, input_source_files_list, training_data_collection, predict_dir, unbuffered=False, model_path=None, max_files=4, inputdir=None):
       
'''
HGCalPredictor(args.inputData, 
               args.dc, 
               args.output_dir, 
               unbuffered=args.unbuffered, 
               max_files=int(args.max_files)).predict(model_path=args.inputModel)





