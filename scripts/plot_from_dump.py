#!/usr/bin/env python3
import gzip

import pickle
import sys

import hplots.hgcal_analysis_plotter_3 as hp

with gzip.open(sys.argv[1], 'rb') as f:
    showers_dataframe, events_dataframe = pickle.load(f)



type = 'hgcal'
if len(sys.argv) == 4:
    type = sys.argv[3]

if type == 'hgcal':
    plotter = hp.HGCalAnalysisPlotter()
elif type =='trackml':
    raise NotImplementedError('ERROR')
    # plotter = tp.TrackMLPlotter()
else:
    raise NotImplementedError("Error")

pdfpath = sys.argv[2]
plotter.set_data(showers_dataframe, events_dataframe, '', pdfpath)
plotter.process()
