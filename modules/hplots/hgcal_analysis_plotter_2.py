import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import experiment_database_reading_manager
from hplots.general_hist_plot import GeneralHistogramPlot
import matching_and_analysis
from matching_and_analysis import one_hot_encode_id
from matplotlib.backends.backend_pdf import PdfPages

from hplots.pid_plots import ConfusionMatrixPlot, RocCurvesPlot

from hplots.general_2d_plot_extensions import General2dBinningPlot
from hplots.general_2d_plot_extensions_2 import EfficiencyFakeRatePlot, ResponsePlot, ResolutionPlot


class HGCalAnalysisPlotter2:
    def eta_x_transform(self, eta):
        eta = np.abs(eta)
        eta[eta>3] = 3.01
        return eta

    def _add_efficiency_plots(self):
        self.all_plots_config += [
            {
                'id': 'v2_efficiency_fo_truth_energy',
                'class': 'efficiency_simple',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Efficiency',
                'fo': 'energy',
                'title': 'Efficiency',
                'file': 'efficiency',
                'bins': self.energy_bins,
            },
            {
                'id': 'v2_efficiency_fo_pt',
                'class': 'efficiency_simple',
                'x_label': 'pT [GeV]',
                'y_label': 'Efficiency',
                'fo': 'pt',
                'title': 'Efficiency',
                'file': 'efficiency',
                'bins': self.pt_bins,
            },
            {
                'id': 'v2_efficiency_fo_local_shower_energy_fraction',
                'class': 'efficiency_simple',
                'x_label': 'Local shower energy fraction',
                'y_label': 'Efficiency',
                'fo': 'local_shower_energy_fraction',
                'title': 'Efficiency',
                'file': 'efficiency',
                'bins': self.local_shower_fraction_bins,
            },
            {
                'id': 'v2_efficiency_fo_local_shower_energy_fraction_flattened_energy_spectrum',
                'class': 'efficiency_energy_spectrum_flattened',
                'x_label': 'Local shower energy fraction',
                'y_label': 'Efficiency (Flattened energy spectrum)',
                'fo': 'local_shower_energy_fraction',
                'title': 'Efficiency',
                'file': 'efficiency',
                'bins': self.local_shower_fraction_bins,
            },
            {
                'id': 'v2_efficiency_fo_eta',
                'class': 'efficiency_simple',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Efficiency',
                'fo': 'eta',
                'title': 'Efficiency',
                'file': 'efficiency',
                'bins': self.eta_bins,
                'x_transform': self.eta_x_transform
            },
            {
                'id': 'v2_efficiency_fo_eta_flattened_energy_spectrum',
                'class': 'efficiency_energy_spectrum_flattened',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Efficiency (Flattened energy spectrum)',
                'fo': 'eta',
                'title': 'Efficiency',
                'file': 'efficiency',
                'bins': self.eta_bins,
                'x_transform': self.eta_x_transform
            },
        ]

    def _add_fake_rate_plots(self):
        self.all_plots_config += [
            {
                'id': 'v2_fake_rate_fo_energy',
                'class': 'fake_rate_simple',
                'x_label': 'Pred Energy [GeV]',
                'y_label': 'Fake rate',
                'fo': 'energy',
                'title': 'Fake Rate',
                'file': 'fake_rate',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_fake_rate_fo_eta',
                'class': 'fake_rate_simple',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Fake rate',
                'fo': 'dep_eta',
                'title': 'Fake Rate',
                'file': 'fake_rate',
                'bins': self.eta_bins,
                'x_transform': self.eta_x_transform
            },
            {
                'id': 'v2_fake_rate_fo_pt',
                'class': 'fake_rate_simple',
                'x_label': 'pT [GeV]',
                'y_label': 'Fake rate',
                'fo': 'pt',
                'title': 'Fake Rate',
                'file': 'fake_rate',
                'bins': self.pt_bins
            },
        ]

    def _add_response_plots(self):
        self.all_plots_config += [
            {
                'id': 'v2_response_fo_truth_energy',
                'class': 'response_simple',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Response',
                'fo': 'energy',
                'title': 'Response',
                'file': 'response',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_response_fo_pt',
                'class': 'response_simple',
                'x_label': 'Truth pT[GeV]',
                'y_label': 'Response',
                'fo': 'pt',
                'title': 'Response',
                'file': 'response',
                'bins': self.pt_bins
            },
            {
                'id': 'v2_response_eta',
                'class': 'response_simple',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Response',
                'fo': 'eta',
                'title': 'Response',
                'file': 'response',
                'bins': self.eta_bins,
                'x_transform': self.eta_x_transform
            },
            {
                'id': 'v2_response_local_shower_energy_fraction',
                'class': 'response_simple',
                'x_label': 'Local shower energy fraction',
                'y_label': 'Response',
                'fo': 'local_shower_energy_fraction',
                'title': 'Response',
                'file': 'response',
                'bins': self.local_shower_fraction_bins
            },
            {
                'id': 'v2_response_dep_fo_truth_energy',
                'class': 'response_dep',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Response $<E_{pred dep}/E_{true}>$',
                'fo': 'energy',
                'title': 'Response',
                'file': 'response',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_response_dep_fo_pt',
                'class': 'response_dep',
                'x_label': 'Truth pT[GeV]',
                'y_label': 'Response $<E_{pred dep}/E_{true}>$',
                'fo': 'pt',
                'title': 'Response',
                'file': 'response',
                'bins': self.pt_bins
            },
            {
                'id': 'v2_response_dep_eta',
                'class': 'response_dep',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Response $<E_{pred dep}/E_{true}>$',
                'fo': 'eta',
                'title': 'Response',
                'file': 'response',
                'bins': self.eta_bins,
                'x_transform': self.eta_x_transform
            },
            {
                'id': 'v2_response_dep_local_shower_energy_fraction',
                'class': 'response_dep',
                'x_label': 'Local shower energy fraction',
                'y_label': 'Response $<E_{pred dep}/E_{true}>$',
                'fo': 'local_shower_energy_fraction',
                'title': 'Response',
                'file': 'response',
                'bins': self.local_shower_fraction_bins
            },

            {
                'id': 'v2_response_fo_truth_energy_energy_spectrum_flattened',
                'class': 'response_energy_spectrum_flattened',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Response -- Flattened energy spectrum',
                'fo': 'energy',
                'title': 'Response',
                'file': 'response',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_response_fo_pt_energy_spectrum_flattened',
                'class': 'response_energy_spectrum_flattened',
                'x_label': 'Truth pT[GeV]',
                'y_label': 'Response -- Flattened energy spectrum',
                'fo': 'pt',
                'title': 'Response',
                'file': 'response',
                'bins': self.pt_bins
            },
            {
                'id': 'v2_response_eta_energy_spectrum_flattened',
                'class': 'response_energy_spectrum_flattened',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Response -- Flattened energy spectrum',
                'fo': 'eta',
                'title': 'Response',
                'file': 'response',
                'bins': self.eta_bins,
                'x_transform': self.eta_x_transform
            },
            {
                'id': 'v2_response_local_shower_energy_fraction_energy_spectrum_flattened',
                'class': 'response_energy_spectrum_flattened',
                'x_label': 'Local shower energy fraction',
                'y_label': 'Response -- Flattened energy spectrum',
                'fo': 'local_shower_energy_fraction',
                'title': 'Response',
                'file': 'response',
                'bins': self.local_shower_fraction_bins
            },
            {
                'id': 'v2_response_dep_fo_truth_energy_energy_spectrum_flattened',
                'class': 'response_dep_energy_spectrum_flattened',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Response $<E_{pred dep}/E_{true}>$ -- Flattened energy spectrum',
                'fo': 'energy',
                'title': 'Response',
                'file': 'response',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_response_dep_fo_pt_energy_spectrum_flattened',
                'class': 'response_dep_energy_spectrum_flattened',
                'x_label': 'Truth pT[GeV]',
                'y_label': 'Response $<E_{pred dep}/E_{true}>$ -- Flattened energy spectrum',
                'fo': 'pt',
                'title': 'Response',
                'file': 'response',
                'bins': self.pt_bins
            },
            {
                'id': 'v2_response_dep_eta_energy_spectrum_flattened',
                'class': 'response_dep_energy_spectrum_flattened',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Response $<E_{pred dep}/E_{true}>$ -- Flattened energy spectrum',
                'fo': 'eta',
                'title': 'Response',
                'file': 'response',
                'bins': self.eta_bins,
                'x_transform': self.eta_x_transform
            },
            {
                'id': 'v2_response_dep_local_shower_energy_fraction_energy_spectrum_flattened',
                'class': 'response_dep_energy_spectrum_flattened',
                'x_label': 'Local shower energy fraction',
                'y_label': 'Response $<E_{pred dep}/E_{true}>$ -- Flattened energy spectrum',
                'fo': 'local_shower_energy_fraction',
                'title': 'Response',
                'file': 'response',
                'bins': self.local_shower_fraction_bins
            },
        ]

    def _add_resolution_plots(self):
        self.all_plots_config += [
            {
                'id': 'v2_resolution_fo_truth_energy',
                'class': 'resolution_simple',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Resolution',
                'fo': 'energy',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_resolution_fo_pt',
                'class': 'resolution_simple',
                'x_label': 'pT [GeV]',
                'y_label': 'Resolution',
                'fo': 'pt',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.pt_bins
            },
            {
                'id': 'v2_resolution_fo_eta',
                'class': 'resolution_simple',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Resolution',
                'fo': 'eta',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.eta_bins,
                'x_transform':self.eta_x_transform
            },
            {
                'id': 'v2_resolution_fo_local_shower_energy_fraction',
                'class': 'resolution_simple',
                'x_label': 'Local shower energy fraction',
                'y_label': 'Resolution',
                'fo': 'local_shower_energy_fraction',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.local_shower_fraction_bins
            },
            {
                'id': 'v2_resolution_fo_truth_energy_dep',
                'class': 'resolution_dep',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Resolution wrt $<E_{pred dep}/E_{true}>$',
                'fo': 'energy',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_resolution_fo_pt_dep',
                'class': 'resolution_dep',
                'x_label': 'pT [GeV]',
                'y_label': 'Resolution wrt $<E_{pred dep}/E_{true}>$',
                'fo': 'pt',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.pt_bins
            },
            {
                'id': 'v2_resolution_fo_eta_dep',
                'class': 'resolution_dep',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Resolution wrt $<E_{pred dep}/E_{true}>$',
                'fo': 'eta',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.eta_bins,
                'x_transform':self.eta_x_transform
            },
            {
                'id': 'v2_resolution_fo_local_shower_energy_fraction_dep',
                'class': 'resolution_dep',
                'x_label': 'Local shower energy fraction',
                'y_label': 'Resolution wrt $<E_{pred dep}/E_{true}>$',
                'fo': 'local_shower_energy_fraction',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.local_shower_fraction_bins
            },
            {
                'id': 'v2_response_fo_pred_energy',
                'class': 'response_pred',
                'x_label': 'Pred Energy [GeV]',
                'y_label': 'Response',
                'fo': 'energy',
                'title': 'Response',
                'file': 'response',
                'bins': self.energy_bins
            },
        ]

    def _add_plots(self):
        self.all_plots_config = []
        self._add_efficiency_plots()
        self._add_fake_rate_plots()
        self._add_response_plots()
        self._add_resolution_plots()


    def __init__(self,log_of_distributions=True):
        self.energy_bins = np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200])
        self.local_shower_fraction_bins = np.array([0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        self.eta_bins = np.array([1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.25,2.5,2.75,3,3.1])
        self.pt_bins = np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80])


        self._add_plots()
        self._build_plotter_classes()



    def _build_plotter_classes(self):
        self.all_plots = []
        for p in self.all_plots_config:
            if p['class'] == 'efficiency_simple' or p['class']=='fake_rate_simple' or p['class'] == 'efficiency_energy_spectrum_flattened':
                plot = EfficiencyFakeRatePlot(
                    bins=p['bins'],
                    x_label=p['x_label'],
                    y_label=p['y_label'],
                    title=p['title'],
                )
                self.all_plots.append(plot)
            elif p['class'] == 'response_simple'\
                    or p['class'] =='response_dep'\
                    or p['class'] =='response_dep_energy_spectrum_flattened'\
                    or p['class'] == 'response_energy_spectrum_flattened'\
                    or p['class'] == 'response_pred':
                plot = ResponsePlot(
                    bins=p['bins'],
                    x_label=p['x_label'],
                    y_label=p['y_label'],
                    title=p['title'],
                )
                self.all_plots.append(plot)
            elif p['class'] == 'resolution_simple' or p['class'] == 'resolution_dep':
                plot = ResolutionPlot(
                    bins=p['bins'],
                    x_label=p['x_label'],
                    y_label=p['y_label'],
                    title=p['title'],
                )
                self.all_plots.append(plot)
            else:
                raise NotImplementedError()





    def _draw_numerics(self):
        text_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal',
                     'verticalalignment': 'bottom'}
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_visible(False)
        ax.axis('off')

        bs = ','.join(['%.5f'%x for x in self.beta_thresholds])
        ds = ','.join(['%.5f'%x for x in self.dist_thresholds])
        iss = ','.join(['%.5f'%x for x in self.iou_thresholds])
        matching_types = ','.join([str(x) for x in self.matching_types])

        eprecisions = self.pred_energy_matched
        erecalls = self.truth_energy_matched
        fscores = self.reco_scores


        if len(self.pred_energy_matched) == 0:
            eprecisions = self.pred_energy_matched + [-1]
        if len(self.truth_energy_matched) == 0:
            erecalls = self.truth_energy_matched + [-1]
        if len(self.reco_scores) == 0:
            fscores = self.reco_scores + [-1]
        sp = ','.join(['%.5f'%x for x in eprecisions])
        sr = ','.join(['%.5f'%x for x in erecalls])
        sf = ','.join(['%.5f'%x for x in fscores])

        s = 'Beta threshold: %s\nDist threshold: %s\niou  threshold: %s\nMatching types: %s\n' \
            "%% pred energy matched: %s\n%% truth energy matched: %s\nReco score: %s" % (bs, ds, iss, matching_types, sp, sr, sf)

        plt.text(0, 1, s, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                 fontdict=text_font)

    def write_data_to_database(self, database_manager, table_prefix):
        pass


        database_manager.flush()

    def add_data_from_database(self, database_reading_manager, table_prefix, experiment_name=None, condition=None):
        pass

    def _get_tags(self, metadata, label, additional_tags):
        tags = dict()
        tags['beta_threshold'] = float(metadata['beta_threshold'])
        tags['distance_threshold'] = float(metadata['distance_threshold'])
        tags['iou_threshold'] = float(metadata['iou_threshold'])
        tags['matching_type'] = str(metadata['matching_type_str'])
        tags['label'] = str(label)

        skip = {'beta_threshold', 'distance_threshold', 'iou_threshold', 'matching_type', 'label', 'matching_type_str'}

        for key, value in metadata.items():
            if key in skip:
                continue
            if type(value) is float or type(value) is int:
                if np.isfinite(value):
                    tags[key] = value
            if type(value) is str:
                if len(value) < 100:
                    tags[key] = value

        for key, value in additional_tags.items():
            tags[key] = value

    def add_data_from_analysed_graph_list(self, analysed_graphs, metadata, label='', additional_tags=dict()):
        tags = self._get_tags(metadata, label, additional_tags)

        for plot,config in zip(self.all_plots, self.all_plots_config):
            if config['class']=='efficiency_simple':
                x, y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, config['fo'], 'energy',
                                                                         numpy=True, not_found_value=-1, sum_multi=True)
                y = np.not_equal(y, -1)
                if 'x_transform' in config:
                    x = config['x_transform'](x)

                plot.add_raw_values(x, y, tags)

            elif config['class']=='efficiency_energy_spectrum_flattened':
                truth_values, pred_values = matching_and_analysis.get_truth_matched_attributes(
                    analysed_graphs, {config['fo'],'energy'}, {'energy'},
                    numpy=True, not_found_value=-1)

                x = truth_values[config['fo']]
                et = truth_values['energy']
                ep = pred_values['energy']

                bins = self.energy_bins
                freq = []
                for i in range(len(bins) - 1):
                    l = bins[i]
                    h = bins[i + 1]
                    filter = np.logical_and(et >= l, et < h)
                    s = float(np.sum(filter)) / float((h - l))
                    freq.append(s)
                z = np.searchsorted(bins, et) - 1
                z = np.minimum(np.maximum(z, 0), len(freq) - 1)
                weights = np.array([1. / freq[x] for x in z])

                if 'x_transform' in config:
                    x = config['x_transform'](x)

                filter = np.not_equal(ep, -1)

                plot.add_raw_values(x, filter, tags, weights=weights)

            elif config['class']=='fake_rate_simple':
                x, y = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, config['fo'], 'energy',
                                                                         numpy=True, not_found_value=-1, sum_multi=True)
                y = np.equal(y, -1)
                if 'x_transform' in config:
                    x = config['x_transform'](x)

                plot.add_raw_values(x, y, tags)
            elif config['class']=='response_simple' or config['class']=='resolution_simple' or config['class'] == 'response_energy_spectrum_flattened':
                truth_values, pred_values = matching_and_analysis.get_truth_matched_attributes(
                    analysed_graphs, {config['fo'],'energy'}, {'energy'},
                    numpy=True, not_found_value=-1)

                x = truth_values[config['fo']]
                et = truth_values['energy']
                ep = pred_values['energy']

                bins = self.energy_bins
                freq = []
                for i in range(len(bins) - 1):
                    l = bins[i]
                    h = bins[i + 1]
                    filter = np.logical_and(et >= l, et < h)
                    s = float(np.sum(filter)) / float((h - l))
                    freq.append(s)
                z = np.searchsorted(bins, et) - 1
                z = np.minimum(np.maximum(z, 0), len(freq) - 1)
                weights = np.array([1. / freq[x] for x in z])

                if 'x_transform' in config:
                    x = config['x_transform'](x)

                filter = np.not_equal(ep, -1)
                y = ep[filter]/et[filter]

                plot.add_raw_values(x[filter], y, tags, weights=weights if config['class']=='response_energy_spectrum_flattened' else None)
            elif config['class'] == 'response_pred':
                pred_values, truth_values = matching_and_analysis.get_pred_matched_attributes(
                    analysed_graphs, {config['fo'], 'energy'}, {'energy'},
                    numpy=True, not_found_value=-1)

                x = pred_values[config['fo']]
                ep = pred_values['energy']
                et = truth_values['energy']

                if 'x_transform' in config:
                    x = config['x_transform'](x)

                filter = np.not_equal(et, -1)

                y = ep[filter] / et[filter]

                plot.add_raw_values(x[filter], y, tags)

            elif config['class']=='response_dep' or config['class']=='resolution_dep' or config['class'] == 'response_dep_energy_spectrum_flattened':
                truth_values, pred_values = matching_and_analysis.get_truth_matched_attributes(
                    analysed_graphs, {config['fo'],'energy'}, {'dep_energy'},
                    numpy=True, not_found_value=-1)

                x = truth_values[config['fo']]
                et = truth_values['energy']
                ep = pred_values['dep_energy']
                bins = self.energy_bins
                freq = []
                for i in range(len(bins) - 1):
                    l = bins[i]
                    h = bins[i + 1]
                    filter = np.logical_and(et >= l, et < h)
                    s = float(np.sum(filter)) / float((h - l))
                    freq.append(s)
                z = np.searchsorted(bins, et) - 1
                z = np.minimum(np.maximum(z, 0), len(freq) - 1)
                weights = np.array([1. / freq[x] for x in z])

                if 'x_transform' in config:
                    x = config['x_transform'](x)

                filter = np.not_equal(ep, -1)
                y = ep[filter]/et[filter]

                plot.add_raw_values(x[filter], y, tags, weights=weights if config['class']=='response_dep_energy_spectrum_flattened' else None)
            else:
                print(config['class'])
                raise NotImplementedError()

    def write_to_pdf(self, pdfpath, formatter=lambda x:''):
        if os.path.exists(pdfpath):
            if os.path.isdir(pdfpath):
                shutil.rmtree(pdfpath)
            else:
                os.unlink(pdfpath)

        os.mkdir(pdfpath)

        pdf_efficiency = PdfPages(os.path.join(pdfpath,'efficiency.pdf'))
        pdf_response = PdfPages(os.path.join(pdfpath,'response.pdf'))
        pdf_pid = PdfPages(os.path.join(pdfpath,'pid.pdf'))
        pdf_fake_rate = PdfPages(os.path.join(pdfpath,'fake_rate.pdf'))
        pdf_others = PdfPages(os.path.join(pdfpath,'others.pdf'))
        pdf_resolution = PdfPages(os.path.join(pdfpath,'resolution.pdf'))
        pdf_response_histos = PdfPages(os.path.join(pdfpath,'response_histos.pdf'))

        pdf_writer = {
            'efficiency':pdf_efficiency,
            'response':pdf_response,
            'pid':pdf_pid,
            'fake_rate':pdf_fake_rate,
            'others':pdf_others,
            'resolution':pdf_resolution,
            'response_histos':pdf_response_histos,
        }

        for plot, config in zip(self.all_plots, self.all_plots_config):
            fig = plot.draw(formatter)
            pdf_writer[config['file']].savefig(fig)



        pdf_efficiency.close()
        pdf_response.close()
        pdf_fake_rate.close()
        pdf_others.close()
        pdf_pid.close()
        pdf_resolution.close()
        pdf_response_histos.close()

        plt.close('all')
