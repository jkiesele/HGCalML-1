import os
import shutil

from matplotlib.backends.backend_pdf import PdfPages

from hplots.general_2d_plot_extensions import EfficiencyFoTruthEnergyPlot, ResolutionFoEnergyPlot, ResolutionFoTruthEta, \
    ResolutionFoLocalShowerEnergyFraction
from hplots.general_2d_plot_extensions import EfficiencyFoTruthPIDPlot
from hplots.general_2d_plot_extensions import ResponseFoTruthPIDPlot

import hplots.general_2d_plot_extensions_2 as hp2


from hplots.general_hist_extensions import ResponseHisto, Multi4HistEnergy, Multi4HistPt

import numpy as np
import matplotlib.pyplot as plt
import experiment_database_reading_manager
from hplots.general_hist_plot import GeneralHistogramPlot
import matching_and_analysis
from matching_and_analysis import one_hot_encode_id

from hplots.pid_plots import ConfusionMatrixPlot, RocCurvesPlot

def setstyles(fontsize):
    import matplotlib
    axes = {'labelsize': fontsize,
            'titlesize': fontsize}

    matplotlib.rc('axes', **axes)
    matplotlib.rc('legend', fontsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('ytick', labelsize=fontsize)



class HGCalAnalysisPlotter:
    def __init__(self, plots = None,log_of_distributions=True):

        self.energy_bins = np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200])
        self.local_shower_fraction_bins = np.array([0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        self.eta_bins = np.array([1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.25,2.5,2.75,3,3.1])
        self.pt_bins = np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80])

        self.total_response_bins = [0.79,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,
                                    1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.2,1.21]

        # self.e_other_bins= [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8,8.25,8.5,8.75,9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,11.25,11.5,11.75,12,12.25,12.5,12.75,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.25,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,19.75,20]
        self.e_other_bins = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0, 23.25, 23.5, 23.75, 24.0, 24.25, 24.5, 24.75, 25.0, 25.25, 25.5, 25.75, 26.0, 26.25, 26.5, 26.75, 27.0, 27.25, 27.5, 27.75, 28.0, 28.25, 28.5, 28.75, 29.0, 29.25, 29.5, 29.75, 30.0, 30.25, 30.5, 30.75, 31.0, 31.25, 31.5, 31.75, 32.0, 32.25, 32.5, 32.75, 33.0, 33.25, 33.5, 33.75, 34.0, 34.25, 34.5, 34.75, 35.0, 35.25, 35.5, 35.75, 36.0, 36.25, 36.5, 36.75, 37.0, 37.25, 37.5, 37.75, 38.0, 38.25, 38.5, 38.75, 39.0, 39.25, 39.5, 39.75, 40.0, 40.25, 40.5, 40.75, 41.0, 41.25, 41.5, 41.75, 42.0, 42.25, 42.5, 42.75, 43.0, 43.25, 43.5, 43.75, 44.0, 44.25, 44.5, 44.75, 45.0, 45.25, 45.5, 45.75, 46.0, 46.25, 46.5, 46.75, 47.0, 47.25, 47.5, 47.75, 48.0, 48.25, 48.5, 48.75, 49.0, 49.25, 49.5, 49.75, 50.0, 50.25, 50.5, 50.75, 51.0, 51.25, 51.5, 51.75, 52.0, 52.25, 52.5, 52.75, 53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75, 55.0, 55.25, 55.5, 55.75, 56.0, 56.25, 56.5, 56.75, 57.0, 57.25, 57.5, 57.75, 58.0, 58.25, 58.5, 58.75, 59.0, 59.25, 59.5, 59.75, 60.0, 60.25, 60.5, 60.75, 61.0, 61.25, 61.5, 61.75, 62.0, 62.25, 62.5, 62.75, 63.0, 63.25, 63.5, 63.75, 64.0, 64.25, 64.5, 64.75, 65.0, 65.25, 65.5, 65.75, 66.0, 66.25, 66.5, 66.75, 67.0, 67.25, 67.5, 67.75, 68.0, 68.25, 68.5, 68.75, 69.0, 69.25, 69.5, 69.75, 70.0, 70.25, 70.5, 70.75, 71.0, 71.25, 71.5, 71.75, 72.0, 72.25, 72.5, 72.75, 73.0, 73.25, 73.5, 73.75, 74.0, 74.25, 74.5, 74.75, 75.0, 75.25, 75.5, 75.75, 76.0, 76.25, 76.5, 76.75, 77.0, 77.25, 77.5, 77.75, 78.0, 78.25, 78.5, 78.75, 79.0, 79.25, 79.5, 79.75, 80.0, 80.25, 80.5, 80.75, 81.0, 81.25, 81.5, 81.75, 82.0, 82.25, 82.5, 82.75, 83.0, 83.25, 83.5, 83.75, 84.0, 84.25, 84.5, 84.75, 85.0, 85.25, 85.5, 85.75, 86.0, 86.25, 86.5, 86.75, 87.0, 87.25, 87.5, 87.75, 88.0, 88.25, 88.5, 88.75, 89.0, 89.25, 89.5, 89.75, 90.0, 90.25, 90.5, 90.75, 91.0, 91.25, 91.5, 91.75, 92.0, 92.25, 92.5, 92.75, 93.0, 93.25, 93.5, 93.75, 94.0, 94.25, 94.5, 94.75, 95.0, 95.25, 95.5, 95.75, 96.0, 96.25, 96.5, 96.75, 97.0, 97.25, 97.5, 97.75, 98.0, 98.25, 98.5, 98.75, 99.0, 99.25, 99.5, 99.75, 100.0, 100.25, 100.5, 100.75, 101.0, 101.25, 101.5, 101.75, 102.0, 102.25, 102.5, 102.75, 103.0, 103.25, 103.5, 103.75, 104.0, 104.25, 104.5, 104.75, 105.0, 105.25, 105.5, 105.75, 106.0, 106.25, 106.5, 106.75, 107.0, 107.25, 107.5, 107.75, 108.0, 108.25, 108.5, 108.75, 109.0, 109.25, 109.5, 109.75, 110.0, 110.25, 110.5, 110.75, 111.0, 111.25, 111.5, 111.75, 112.0, 112.25, 112.5, 112.75, 113.0, 113.25, 113.5, 113.75, 114.0, 114.25, 114.5, 114.75, 115.0, 115.25, 115.5, 115.75, 116.0, 116.25, 116.5, 116.75, 117.0, 117.25, 117.5, 117.75, 118.0, 118.25, 118.5, 118.75, 119.0, 119.25, 119.5, 119.75, 120.0, 120.25, 120.5, 120.75, 121.0, 121.25, 121.5, 121.75, 122.0, 122.25, 122.5, 122.75, 123.0, 123.25, 123.5, 123.75, 124.0, 124.25, 124.5, 124.75, 125.0, 125.25, 125.5, 125.75, 126.0, 126.25, 126.5, 126.75, 127.0, 127.25, 127.5, 127.75, 128.0, 128.25, 128.5, 128.75, 129.0, 129.25, 129.5, 129.75, 130.0, 130.25, 130.5, 130.75, 131.0, 131.25, 131.5, 131.75, 132.0, 132.25, 132.5, 132.75, 133.0, 133.25, 133.5, 133.75, 134.0, 134.25, 134.5, 134.75, 135.0, 135.25, 135.5, 135.75, 136.0, 136.25, 136.5, 136.75, 137.0, 137.25, 137.5, 137.75, 138.0, 138.25, 138.5, 138.75, 139.0, 139.25, 139.5, 139.75, 140.0, 140.25, 140.5, 140.75, 141.0, 141.25, 141.5, 141.75, 142.0, 142.25, 142.5, 142.75, 143.0, 143.25, 143.5, 143.75, 144.0, 144.25, 144.5, 144.75, 145.0, 145.25, 145.5, 145.75, 146.0, 146.25, 146.5, 146.75, 147.0, 147.25, 147.5, 147.75, 148.0, 148.25, 148.5, 148.75, 149.0, 149.25, 149.5, 149.75, 150.0, 150.25, 150.5, 150.75, 151.0, 151.25, 151.5, 151.75, 152.0, 152.25, 152.5, 152.75, 153.0, 153.25, 153.5, 153.75, 154.0, 154.25, 154.5, 154.75, 155.0, 155.25, 155.5, 155.75, 156.0, 156.25, 156.5, 156.75, 157.0, 157.25, 157.5, 157.75, 158.0, 158.25, 158.5, 158.75, 159.0, 159.25, 159.5, 159.75, 160.0, 160.25, 160.5, 160.75, 161.0, 161.25, 161.5, 161.75, 162.0, 162.25, 162.5, 162.75, 163.0, 163.25, 163.5, 163.75, 164.0, 164.25, 164.5, 164.75, 165.0, 165.25, 165.5, 165.75, 166.0, 166.25, 166.5, 166.75, 167.0, 167.25, 167.5, 167.75, 168.0, 168.25, 168.5, 168.75, 169.0, 169.25, 169.5, 169.75, 170.0, 170.25, 170.5, 170.75, 171.0, 171.25, 171.5, 171.75, 172.0, 172.25, 172.5, 172.75, 173.0, 173.25, 173.5, 173.75, 174.0, 174.25, 174.5, 174.75, 175.0, 175.25, 175.5, 175.75, 176.0, 176.25, 176.5, 176.75, 177.0, 177.25, 177.5, 177.75, 178.0, 178.25, 178.5, 178.75, 179.0, 179.25, 179.5, 179.75, 180.0, 180.25, 180.5, 180.75, 181.0, 181.25, 181.5, 181.75, 182.0, 182.25, 182.5, 182.75, 183.0, 183.25, 183.5, 183.75, 184.0, 184.25, 184.5, 184.75, 185.0, 185.25, 185.5, 185.75, 186.0, 186.25, 186.5, 186.75, 187.0, 187.25, 187.5, 187.75, 188.0, 188.25, 188.5, 188.75, 189.0, 189.25, 189.5, 189.75, 190.0, 190.25, 190.5, 190.75, 191.0, 191.25, 191.5, 191.75, 192.0, 192.25, 192.5, 192.75, 193.0, 193.25, 193.5, 193.75, 194.0, 194.25, 194.5, 194.75, 195.0, 195.25, 195.5, 195.75, 196.0, 196.25, 196.5, 196.75, 197.0, 197.25, 197.5, 197.75, 198.0, 198.25, 198.5, 198.75, 199.0, 199.25, 199.5, 199.75,200]


        self.efficiency_fo_truth_pid_plot = EfficiencyFoTruthPIDPlot(histogram_log=log_of_distributions)
        self.response_fo_truth_pid_plot = ResponseFoTruthPIDPlot(histogram_log=log_of_distributions)
        self.confusion_matrix_plot = ConfusionMatrixPlot()
        self.roc_curves = RocCurvesPlot()

        self.response_histogam = ResponseHisto()
        self.response_histogam_divided = Multi4HistEnergy()

        self.response_pt_histogam = ResponseHisto(x_label='${p_T}_{true}/{p_T}_{pred}$')
        self.response_pt_histogam_divided = Multi4HistPt()

        self.total_dep_to_impact = ResponseHisto(
            x_label='$\\frac{x}{\\sum E_{true}}$',
            y_label='Frequency',
            bins=np.array(self.total_response_bins),
        )

        # self.total_pred_to_impact = ResponseHisto(
        #     x_label='$\\frac{\\sum E_{pred}}{\\sum E_{true}}$',
        #     y_label='Frequency',
        #     bins=np.array(self.total_response_bins),
        # )

        self.total_no_noise_to_pred = ResponseHisto(
            x_label='$\\frac{\\sum E_{dep\_pred\_no\_noise}}{\\sum E_{pred}}$',
            y_label='Frequency',
            bins=np.linspace(0.95,1,15),
        )
        self.total_only_noise_to_pred = ResponseHisto(
            x_label='$\\frac{\\sum E_{dep\_pred\_only\_noise}}{\\sum E_{pred}}$',
            y_label='Frequency',
            bins=np.linspace(0,0.05,15),
        )

        binsx = np.linspace(0,200,801)
        self.e_other_histogram = GeneralHistogramPlot(
            # bins=np.array(self.e_other_bins+[20.25]),
            bins=binsx,
            x_label='E other',
            histogram_log=True
        )

        self.corr_factor_histogram = GeneralHistogramPlot(
            # bins=np.array(self.e_other_bins+[20.25]),
            bins=np.linspace(0.7,1.5, 50),
            x_label='corr factor',
            histogram_log=True
        )

        self.noise_assigned_to_pred_histogram = GeneralHistogramPlot(
            # bins=np.array(self.e_other_bins+[20.25]),
            bins=None,
            x_label='Hit energy sum of noise assigned to pred [GeV]',
            histogram_log=True
        )

        self.noise_assigned_to_pred_to_total_noise_histogram = GeneralHistogramPlot(
            # bins=np.array(self.e_other_bins+[20.25]),
            bins=None,
            x_label='Hit energy sum of noise assigned to pred / hit energy sum of all noise',
            histogram_log=True
        )

        self.resolution_fo_true_energy = ResolutionFoEnergyPlot()
        self.resolution_fo_true_eta = ResolutionFoTruthEta()
        self.resolution_fo_local_shower_energy_fraction = ResolutionFoLocalShowerEnergyFraction()

        self.resolution_sum_fo_true_energy = ResolutionFoEnergyPlot(y_label='Resolution (truth, dep pred)')
        self.resolution_sum_fo_true_eta = ResolutionFoTruthEta(y_label='Resolution (truth, dep pred)')
        self.resolution_sum_fo_local_shower_energy_fraction = ResolutionFoLocalShowerEnergyFraction(y_label='Resolution (truth, dep pred)')

        self.dist_thresholds = []
        self.beta_thresholds = []
        self.iou_thresholds = []
        self.matching_types = []

        plots = [
            'settings',
            'efficiency_fo_truth_pid',
            'response_fo_truth_pid',
            'confusion_matrix',
            'roc_curves',
            'response_histogram',
            'response_histogram_divided',
            'response_pt_histogram',
            'response_pt_histogram_divided',
            'total_response_true_dep_to_impact',
            # 'total_response_pred_to_impact',
            'total_dep_pred_no_noise_to_dep_pred',
            'total_dep_pred_only_noise_to_dep_pred',
            'e_other_histogram',
            'corr_factor_histogram',
            'noise_assigned_to_pred_histogram',
            'noise_assigned_to_pred_to_total_noise_histogram'
        ]

        self.plots = set(plots)
        self.pred_energy_matched = []
        self.truth_energy_matched = []
        self.reco_scores = []

        self._add_plots()
        self._build_plotter_classes()


    def _build_plotter_classes(self):
        self.all_plots = []
        for p in self.all_plots_config:
            if p['class'] == 'efficiency_simple' or p['class']=='fake_rate_simple' or p['class'] == 'efficiency_energy_spectrum_flattened'\
                    or p['class'] == 'efficiency_truth_pu_adjustment':
                plot = hp2.EfficiencyFakeRatePlot(
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
                    or p['class'] == 'response_pred'\
                    or p['class'] == 'response_truth_pu_adjustment':
                plot = hp2.ResponsePlot(
                    bins=p['bins'],
                    x_label=p['x_label'],
                    y_label=p['y_label'],
                    title=p['title'],
                )
                self.all_plots.append(plot)
            elif p['class'] == 'resolution_simple' or p['class'] == 'resolution_dep'\
                    or p['class'] == 'resolution_truth_pu_adjustment':
                plot = hp2.ResolutionPlot(
                    bins=p['bins'],
                    x_label=p['x_label'],
                    y_label=p['y_label'],
                    title=p['title'],
                )
                self.all_plots.append(plot)
            else:
                raise NotImplementedError()


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
                'id': 'v2_efficiency_fo_truth_energy_pu_weighting',
                'class': 'efficiency_truth_pu_adjustment',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Efficiency (redistributed wrt PU)',
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
                'id': 'v2_efficiency_fo_pt_pu_weighting',
                'class': 'efficiency_truth_pu_adjustment',
                'x_label': 'pT [GeV]',
                'y_label': 'Efficiency (redistributed wrt PU)',
                'fo': 'energy',
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
                'id': 'v2_efficiency_fo_eta_pu200',
                'class': 'efficiency_truth_pu_adjustment',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Efficiency (redistributed wrt PU)',
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
                'id': 'v2_response_fo_truth_energy_pu200',
                'class': 'response_truth_pu_adjustment',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Response (redistributed wrt PU)',
                'fo': 'energy',
                'title': 'Response',
                'file': 'response',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_response_fo_e_other',
                'class': 'response_simple',
                'x_label': '$E_{other}$ [GeV]',
                'y_label': 'Response',
                'fo': 'energy_others_in_vicinity',
                'title': 'Response',
                'file': 'response',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_response_fo_e_other_flattened',
                'class': 'response_energy_spectrum_flattened',
                'x_label': '$E_{other}$ [GeV]',
                'y_label': 'Response -- Flattened energy spectrum',
                'fo': 'energy_others_in_vicinity',
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
                'id': 'v2_response_fo_pt_pu200',
                'class': 'response_truth_pu_adjustment',
                'x_label': 'pT [GeV]',
                'y_label': 'Response (redistributed wrt PU)',
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
                'id': 'v2_response_fo_eta_pu200',
                'class': 'response_truth_pu_adjustment',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Response (redistributed wrt PU)',
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
                'id': 'v2_resolution_fo_truth_energy_pu200',
                'class': 'resolution_truth_pu_adjustment',
                'x_label': 'Truth Energy [GeV]',
                'y_label': 'Resolution (redistributed wrt PU)',
                'fo': 'energy',
                'title': 'Resolution',
                'file': 'resolution',
                'bins': self.energy_bins
            },
            {
                'id': 'v2_resolution_fo_e_other',
                'class': 'resolution_simple',
                'x_label': '$E_{other}$ [GeV]',
                'y_label': 'Resolution',
                'fo': 'energy_others_in_vicinity',
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
                'id': 'v2_resolution_fo_pt_pu200',
                'class': 'resolution_truth_pu_adjustment',
                'x_label': 'pT [GeV]',
                'y_label': 'Resolution (redistributed wrt PU)',
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
                'id': 'v2_resolution_fo_eta_pu200',
                'class': 'resolution_truth_pu_adjustment',
                'x_label': '$|\\eta_{true}|$',
                'y_label': 'Resolution (redistributed wrt PU)',
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
                'x_transform':self.eta_x_transform,
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
            {
                'id': 'v2_response_fo_pred_energy_e_true_cut_25',
                'class': 'response_pred',
                'x_label': 'Pred Energy [GeV]',
                'y_label': 'Response -- only for showers with E_true > 25',
                'fo': 'energy',
                'title': 'Response',
                'file': 'response',
                'truth_min_cut_on':'energy',
                'truth_min_cut_value':25.,
                'bins': self.energy_bins
            },
        ]

    def _add_plots(self):
        self.all_plots_config = []
        self._add_efficiency_plots()
        self._add_fake_rate_plots()
        self._add_response_plots()
        self._add_resolution_plots()

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

        return fig

    def write_data_to_database(self, database_manager, table_prefix):
        # self.efficiency_fo_truth_pid_plot.write_to_database(database_manager, table_prefix+'_efficiency_fo_truth_pid')
        # self.response_fo_truth_pid_plot.write_to_database(database_manager, table_prefix+'_response_fo_truth_pid')
        # self.confusion_matrix_plot.write_to_database(database_manager, table_prefix+'_confusion_matrix')
        # database_manager.flush()
       pass

    def add_data_from_database(self, database_reading_manager, table_prefix, experiment_name=None, condition=None):
        return
        # try:
        #     self.response_fo_local_shower_energy_fraction.read_from_database(database_reading_manager, table_prefix + '_response_fo_local_shower_energy_fraction', experiment_name=experiment_name, condition=condition)
        # except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
        #     print("Skipping response_fo_local_shower_energy_fraction, table doesn't exist")
        #
        #
        # tags = self.efficiency_plot.get_tags()
        #
        # self.beta_thresholds += [x['beta_threshold'] for x in tags]
        # self.dist_thresholds += [x['distance_threshold'] for x in tags]
        # self.iou_thresholds += [x['iou_threshold'] for x in tags]
        # self.soft += [x['soft'] for x in tags]
        #
        # self.beta_thresholds = np.unique(self.beta_thresholds).tolist()
        # self.dist_thresholds = np.unique(self.dist_thresholds).tolist()
        # self.iou_thresholds = np.unique(self.iou_thresholds).tolist()
        # self.soft = np.unique(self.soft).tolist()
        #
        #
        # if 'reco_score' in tags[0]:
        #     self.reco_scores += [x['reco_score'] for x in tags]
        #     self.reco_scores = np.unique(self.reco_scores).tolist()
        #
        #
        #     self.pred_energy_matched += [x['pred_energy_percentage_matched'] for x in tags]
        #     self.pred_energy_matched = np.unique(self.pred_energy_matched).tolist()
        #
        #     self.truth_energy_matched += [x['truth_energy_percentage_matched'] for x in tags]
        #     self.truth_energy_matched = np.unique(self.truth_energy_matched).tolist()

    def compute_truth_list_weighting_for_PU(self, graphs):
        # Just computing energy weights for weighing for PU.
        x, _ = matching_and_analysis.get_truth_matched_attribute(graphs, 'energy_others_in_vicinity', 'energy',
                                                                 numpy=True, not_found_value=-1, sum_multi=True)
        e, _ = matching_and_analysis.get_truth_matched_attribute(graphs, 'energy', 'energy',
                                                                 numpy=True, not_found_value=-1, sum_multi=True)

        # plt.scatter(e, x, s=0.1)
        # plt.xlabel('Energy')
        # plt.ylabel('Energy others')
        # plt.show()

        e_my_bins = self.e_other_bins + [np.max(x)+1000]
        weights_dataset,_ = np.histogram(x, e_my_bins)

        weights_pu = np.array([0.33988, 0.390739, 0.389752, 0.357772, 0.313299, 0.267537, 0.225785, 0.189848, 0.159815, 0.135067, 0.114782, 0.0981597, 0.0844993, 0.0732215, 0.0638583, 0.0560371, 0.0494629, 0.0439018, 0.0391687, 0.0351163, 0.0316267, 0.0286053, 0.0259757, 0.0236757, 0.0216547, 0.0198709, 0.0182899, 0.0168832, 0.0156267, 0.0145006, 0.0134878, 0.0125741, 0.0117473, 0.010997, 0.0103143, 0.00969149, 0.00912189, 0.00859977, 0.00812009, 0.00767847, 0.00727107, 0.00689451, 0.0065458, 0.00622232, 0.00592174, 0.00564197, 0.00538118, 0.0051377, 0.00491007, 0.00469696, 0.00449717, 0.00430964, 0.0041334, 0.00396757, 0.00381136, 0.00366404, 0.00352497, 0.00339355, 0.00326923, 0.00315152, 0.00303996, 0.00293414, 0.00283367, 0.0027382, 0.00264741, 0.002561, 0.0024787, 0.00240026, 0.00232543, 0.00225401, 0.00218578, 0.00212057, 0.00205821, 0.00199852, 0.00194136, 0.00188659, 0.00183408, 0.00178371, 0.00173536, 0.00168894, 0.00164433, 0.00160145, 0.0015602, 0.00152052, 0.00148231, 0.00144551, 0.00141006, 0.00137588, 0.00134291, 0.00131111, 0.00128041, 0.00125077, 0.00122213, 0.00119446, 0.00116772, 0.00114185, 0.00111682, 0.0010926, 0.00106916, 0.00104645, 0.00102446, 0.00100315, 0.000982484, 0.00096245, 0.000943017, 0.000924163, 0.000905865, 0.0008881, 0.000870849, 0.000854093, 0.000837811, 0.000821988, 0.000806606, 0.000791648, 0.0007771, 0.000762946, 0.000749173, 0.000735767, 0.000722715, 0.000710006, 0.000697626, 0.000685565, 0.000673813, 0.000662358, 0.000651191, 0.000640302, 0.000629682, 0.000619323, 0.000609216, 0.000599352, 0.000589725, 0.000580327, 0.00057115, 0.000562188, 0.000553434, 0.000544882, 0.000536525, 0.000528358, 0.000520375, 0.000512571, 0.000504939, 0.000497477, 0.000490177, 0.000483036, 0.000476049, 0.000469212, 0.000462521, 0.00045597, 0.000449557, 0.000443278, 0.000437129, 0.000431106, 0.000425207, 0.000419427, 0.000413763, 0.000408213, 0.000402773, 0.000397441, 0.000392213, 0.000387087, 0.000382061, 0.000377131, 0.000372296, 0.000367553, 0.0003629, 0.000358333, 0.000353853, 0.000349455, 0.000345138, 0.000340901, 0.00033674, 0.000332656, 0.000328644, 0.000324705, 0.000320835, 0.000317034, 0.0003133, 0.000309631, 0.000306026, 0.000302484, 0.000299002, 0.00029558, 0.000292216, 0.000288908, 0.000285657, 0.00028246, 0.000279316, 0.000276223, 0.000273182, 0.000270191, 0.000267248, 0.000264353, 0.000261504, 0.000258701, 0.000255943, 0.000253229, 0.000250557, 0.000247927, 0.000245338, 0.000242789, 0.00024028, 0.000237809, 0.000235377, 0.000232981, 0.000230621, 0.000228297, 0.000226007, 0.000223752, 0.000221531, 0.000219342, 0.000217185, 0.00021506, 0.000212966, 0.000210902, 0.000208868, 0.000206863, 0.000204886, 0.000202938, 0.000201017, 0.000199124, 0.000197257, 0.000195416, 0.0001936, 0.00019181, 0.000190044, 0.000188302, 0.000186585, 0.00018489, 0.000183219, 0.00018157, 0.000179943, 0.000178337, 0.000176753, 0.00017519, 0.000173648, 0.000172126, 0.000170623, 0.000169141, 0.000167677, 0.000166232, 0.000164806, 0.000163398, 0.000162008, 0.000160635, 0.00015928, 0.000157942, 0.000156621, 0.000155316, 0.000154027, 0.000152754, 0.000151497, 0.000150255, 0.000149029, 0.000147817, 0.00014662, 0.000145437, 0.000144269, 0.000143115, 0.000141974, 0.000140847, 0.000139733, 0.000138633, 0.000137545, 0.00013647, 0.000135407, 0.000134357, 0.000133319, 0.000132293, 0.000131279, 0.000130276, 0.000129285, 0.000128305, 0.000127336, 0.000126378, 0.000125431, 0.000124494, 0.000123568, 0.000122652, 0.000121746, 0.00012085, 0.000119964, 0.000119087, 0.00011822, 0.000117363, 0.000116515, 0.000115676, 0.000114846, 0.000114024, 0.000113212, 0.000112408, 0.000111613, 0.000110826, 0.000110047, 0.000109277, 0.000108514, 0.00010776, 0.000107013, 0.000106274, 0.000105543, 0.000104819, 0.000104103, 0.000103394, 0.000102692, 0.000101997, 0.000101309, 0.000100628, 9.99541e-05, 9.92868e-05, 9.86261e-05, 9.7972e-05, 9.73243e-05, 9.6683e-05, 9.60481e-05, 9.54193e-05, 9.47967e-05, 9.41802e-05, 9.35696e-05, 9.29649e-05, 9.23661e-05, 9.1773e-05, 9.11856e-05, 9.06039e-05, 9.00276e-05, 8.94568e-05, 8.88914e-05, 8.83314e-05, 8.77766e-05, 8.7227e-05, 8.66826e-05, 8.61432e-05, 8.56088e-05, 8.50794e-05, 8.45548e-05, 8.40351e-05, 8.35202e-05, 8.30099e-05, 8.25043e-05, 8.20033e-05, 8.15069e-05, 8.10149e-05, 8.05274e-05, 8.00442e-05, 7.95654e-05, 7.90908e-05, 7.86205e-05, 7.81543e-05, 7.76923e-05, 7.72343e-05, 7.67804e-05, 7.63305e-05, 7.58845e-05, 7.54423e-05, 7.50041e-05, 7.45696e-05, 7.41389e-05, 7.37119e-05, 7.32885e-05, 7.28688e-05, 7.24527e-05, 7.20401e-05, 7.16311e-05, 7.12255e-05, 7.08233e-05, 7.04245e-05, 7.00291e-05, 6.9637e-05, 6.92482e-05, 6.88626e-05, 6.84802e-05, 6.8101e-05, 6.77249e-05, 6.73519e-05, 6.6982e-05, 6.66151e-05, 6.62512e-05, 6.58903e-05, 6.55323e-05, 6.51773e-05, 6.48251e-05, 6.44757e-05, 6.41292e-05, 6.37854e-05, 6.34444e-05, 6.31061e-05, 6.27705e-05, 6.24376e-05, 6.21073e-05, 6.17796e-05, 6.14545e-05, 6.11319e-05, 6.08119e-05, 6.04944e-05, 6.01793e-05, 5.98667e-05, 5.95566e-05, 5.92488e-05, 5.89434e-05, 5.86404e-05, 5.83396e-05, 5.80412e-05, 5.77451e-05, 5.74512e-05, 5.71596e-05, 5.68701e-05, 5.65829e-05, 5.62978e-05, 5.60149e-05, 5.57341e-05, 5.54554e-05, 5.51788e-05, 5.49042e-05, 5.46317e-05, 5.43612e-05, 5.40927e-05, 5.38262e-05, 5.35616e-05, 5.3299e-05, 5.30383e-05, 5.27795e-05, 5.25226e-05, 5.22676e-05, 5.20144e-05, 5.1763e-05, 5.15135e-05, 5.12658e-05, 5.10198e-05, 5.07756e-05, 5.05332e-05, 5.02924e-05, 5.00534e-05, 4.98161e-05, 4.95805e-05, 4.93465e-05, 4.91142e-05, 4.88835e-05, 4.86545e-05, 4.8427e-05, 4.82012e-05, 4.79769e-05, 4.77541e-05, 4.75329e-05, 4.73133e-05, 4.70951e-05, 4.68785e-05, 4.66633e-05, 4.64497e-05, 4.62374e-05, 4.60267e-05, 4.58173e-05, 4.56094e-05, 4.54029e-05, 4.51978e-05, 4.49941e-05, 4.47918e-05, 4.45908e-05, 4.43912e-05, 4.41929e-05, 4.39959e-05, 4.38002e-05, 4.36059e-05, 4.34128e-05, 4.3221e-05, 4.30304e-05, 4.28412e-05, 4.26531e-05, 4.24663e-05, 4.22808e-05, 4.20964e-05, 4.19132e-05, 4.17313e-05, 4.15505e-05, 4.13709e-05, 4.11924e-05, 4.10151e-05, 4.08389e-05, 4.06639e-05, 4.049e-05, 4.03172e-05, 4.01454e-05, 3.99748e-05, 3.98053e-05, 3.96369e-05, 3.94695e-05, 3.93031e-05, 3.91379e-05, 3.89736e-05, 3.88104e-05, 3.86482e-05, 3.8487e-05, 3.83269e-05, 3.81677e-05, 3.80095e-05, 3.78523e-05, 3.76961e-05, 3.75408e-05, 3.73865e-05, 3.72331e-05, 3.70807e-05, 3.69292e-05, 3.67786e-05, 3.66289e-05, 3.64802e-05, 3.63324e-05, 3.61854e-05, 3.60394e-05, 3.58942e-05, 3.57499e-05, 3.56065e-05, 3.54639e-05, 3.53222e-05, 3.51813e-05, 3.50413e-05, 3.49021e-05, 3.47637e-05, 3.46261e-05, 3.44894e-05, 3.43535e-05, 3.42183e-05, 3.4084e-05, 3.39505e-05, 3.38177e-05, 3.36857e-05, 3.35545e-05, 3.3424e-05, 3.32943e-05, 3.31654e-05, 3.30372e-05, 3.29097e-05, 3.2783e-05, 3.2657e-05, 3.25318e-05, 3.24072e-05, 3.22834e-05, 3.21603e-05, 3.20378e-05, 3.19161e-05, 3.17951e-05, 3.16747e-05, 3.1555e-05, 3.1436e-05, 3.13177e-05, 3.12001e-05, 3.10831e-05, 3.09667e-05, 3.0851e-05, 3.0736e-05, 3.06216e-05, 3.05078e-05, 3.03947e-05, 3.02822e-05, 3.01703e-05, 3.0059e-05, 2.99484e-05, 2.98383e-05, 2.97289e-05, 2.96201e-05, 2.95118e-05, 2.94042e-05, 2.92971e-05, 2.91906e-05, 2.90847e-05, 2.89794e-05, 2.88747e-05, 2.87705e-05, 2.86668e-05, 2.85638e-05, 2.84613e-05, 2.83593e-05, 2.82579e-05, 2.8157e-05, 2.80567e-05, 2.79569e-05, 2.78576e-05, 2.77588e-05, 2.76606e-05, 2.75629e-05, 2.74657e-05, 2.73691e-05, 2.72729e-05, 2.71772e-05, 2.70821e-05, 2.69874e-05, 2.68933e-05, 2.67996e-05, 2.67064e-05, 2.66137e-05, 2.65215e-05, 2.64297e-05, 2.63385e-05, 2.62477e-05, 2.61573e-05, 2.60675e-05, 2.59781e-05, 2.58891e-05, 2.58007e-05, 2.57126e-05, 2.5625e-05, 2.55379e-05, 2.54512e-05, 2.5365e-05, 2.52791e-05, 2.51938e-05, 2.51088e-05, 2.50243e-05, 2.49402e-05, 2.48565e-05, 2.47732e-05, 2.46904e-05, 2.4608e-05, 2.4526e-05, 2.44444e-05, 2.43632e-05, 2.42824e-05, 2.4202e-05, 2.4122e-05, 2.40424e-05, 2.39632e-05, 2.38844e-05, 2.38059e-05, 2.37279e-05, 2.36502e-05, 2.35729e-05, 2.3496e-05, 2.34195e-05, 2.33434e-05, 2.32676e-05, 2.31922e-05, 2.31171e-05, 2.30424e-05, 2.29681e-05, 2.28941e-05, 2.28205e-05, 2.27473e-05, 2.26744e-05, 2.26018e-05, 2.25296e-05, 2.24577e-05, 2.23862e-05, 2.2315e-05, 2.22442e-05, 2.21737e-05, 2.21035e-05, 2.20337e-05, 2.19642e-05, 2.1895e-05, 2.18261e-05, 2.17576e-05, 2.16894e-05, 2.16215e-05, 2.15539e-05, 2.14867e-05, 2.14198e-05, 2.13531e-05, 2.12868e-05, 2.12208e-05, 2.11551e-05, 2.10897e-05, 2.10246e-05, 2.09598e-05, 2.08953e-05, 2.08311e-05, 2.07672e-05, 2.07036e-05, 2.06403e-05, 2.05772e-05, 2.05145e-05, 2.0452e-05, 2.03899e-05, 2.0328e-05, 2.02664e-05, 2.0205e-05, 2.0144e-05, 2.00832e-05, 2.00227e-05, 1.99625e-05, 1.99025e-05, 1.98428e-05, 1.97834e-05, 1.97243e-05, 1.96654e-05, 1.96067e-05, 1.95484e-05, 1.94903e-05, 1.94324e-05, 1.93748e-05, 1.93175e-05, 1.92604e-05, 1.92036e-05, 1.9147e-05, 1.90907e-05, 1.90346e-05, 1.89788e-05, 1.89232e-05, 1.88679e-05, 1.88128e-05, 1.87579e-05, 1.87033e-05, 1.86489e-05, 1.85947e-05, 1.85408e-05, 1.84871e-05, 1.84337e-05, 1.83805e-05, 1.83275e-05, 1.82747e-05, 1.82222e-05, 1.81699e-05, 1.81178e-05, 1.8066e-05, 1.80144e-05, 1.79629e-05, 1.79117e-05, 1.78608e-05, 1.781e-05, 1.77595e-05, 1.77091e-05, 1.7659e-05, 1.76091e-05, 1.75594e-05, 1.751e-05, 1.74607e-05, 1.74116e-05, 1.73628e-05, 1.73141e-05, 1.72657e-05, 1.72174e-05, 1.71694e-05, 1.71215e-05, 1.70739e-05, 1.70265e-05, 1.69792e-05, 1.69322e-05, 1.68853e-05, 1.68386e-05, 1.67922e-05, 1.67459e-05, 1.66998e-05, 1.66539e-05, 1.66082e-05, 1.65627e-05, 1.65173e-05, 1.64722e-05, 1.64272e-05, 1.63824e-05, 1.63379e-05, 1.62934e-05, 1.62492e-05, 1.62051e-05, 1.61613e-05, 1.61176e-05, 1.60741e-05, 1.60307e-05, 1.59875e-05, 1.59445e-05, 1.59017e-05, 1.58591e-05, 1.58166e-05, 1.57743e-05, 1.57321e-05, 1.56902e-05, 1.56484e-05, 1.56067e-05, 1.55652e-05, 1.55239e-05, 1.54828e-05, 1.54418e-05, 1.5401e-05, 1.53603e-05, 1.53198e-05, 1.52795e-05, 1.52393e-05, 1.51993e-05, 1.51595e-05, 1.51198e-05, 1.50802e-05, 1.50408e-05, 1.50016e-05, 1.49625e-05, 1.49235e-05, 1.48848e-05, 1.48461e-05, 1.48076e-05, 1.47693e-05, 1.47311e-05, 1.46931e-05, 1.46552e-05, 1.46175e-05, 1.45799e-05, 1.45424e-05, 1.45051e-05, 1.44679e-05, 1.44309e-05, 1.4394e-05, 1.43573e-05, 1.43207e-05, 1.42842e-05, 1.42479e-05, 1.42117e-05, 1.41757e-05, 1.41398e-05, 0])
        weights_calc = weights_pu / weights_dataset

        z = np.searchsorted(e_my_bins, x) - 1
        weights = [weights_calc[x] for x in z]

        return np.array(weights)

    def add_data_from_analysed_graph_list(self, analysed_graphs, metadata, label='', additional_tags=dict()):
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

        self.beta_thresholds.append(float(tags['beta_threshold']))
        self.dist_thresholds.append(float(tags['distance_threshold']))
        self.iou_thresholds.append(float(tags['iou_threshold']))
        self.matching_types.append(str(metadata['matching_type_str']))

        self.beta_thresholds = np.unique(self.beta_thresholds).tolist()
        self.dist_thresholds = np.unique(self.dist_thresholds).tolist()
        self.iou_thresholds = np.unique(self.iou_thresholds).tolist()

        self.truth_weights_for_pu = self.compute_truth_list_weighting_for_PU(analysed_graphs)

        if 'reco_score' in metadata:
            self.reco_scores.append(metadata['reco_score'])
            self.reco_scores = np.unique(self.reco_scores).tolist()

            self.pred_energy_matched.append(metadata['pred_energy_percentage_matched'])
            self.pred_energy_matched = np.unique(self.pred_energy_matched).tolist()

            self.truth_energy_matched.append(metadata['truth_energy_percentage_matched'])
            self.truth_energy_matched = np.unique(self.truth_energy_matched).tolist()

        x, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'dep_energy', 'energy', numpy=True,
                                                                 not_found_value=-1, sum_multi=True)
        y, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True,
                                                                 not_found_value=-1, sum_multi=True)

        # plt.scatter(x,y, s=0.1)
        # plt.show()

        if 'efficiency_fo_truth_pid' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pid', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            x = np.abs(x)
            x[(x > 29) & (x < 100)] = 29
            x[x==111] = 31
            x[x==211] = 32
            x[x==113] = 33
            x[x==213] = 34
            x[x==115] = 35
            x[x==215] = 36
            x[x==117] = 37
            x[x==217] = 38
            x[x==119] = 39
            x[x==219] = 40
            x[x==130] = 41
            x[x==310] = 42
            x[x==311] = 43
            x[x==321] = 44
            x[x==2212] = 46
            x[x==2112] = 47
            x[x>=100] = 49
            y = y!=-1
            self.efficiency_fo_truth_pid_plot.add_raw_values(x, y, tags)

        if 'response_fo_truth_pid' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pid', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l = np.abs(l)
            l[(l > 29) & (l < 100)] = 29
            l[l==111] = 31
            l[l==211] = 32
            l[l==113] = 33
            l[l==213] = 34
            l[l==115] = 35
            l[l==215] = 36
            l[l==117] = 37
            l[l==217] = 38
            l[l==119] = 39
            l[l==219] = 40
            l[l==130] = 41
            l[l==310] = 42
            l[l==311] = 43
            l[l==321] = 44
            l[l==2212] = 46
            l[l==2112] = 47
            l[l>=100] = 49
            filter = y!=-1
            self.response_fo_truth_pid_plot.add_raw_values(l[filter], y[filter] / x[filter], tags)

        if 'confusion_matrix' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pid', 'pid_probability', numpy=False, not_found_value=None, sum_multi=True)

            filter = np.argwhere(np.array([a is not None for a in y], np.bool))
            filter = filter[:, 0]

            x = [x[i] for i in filter]
            y = [y[i] for i in filter]

            x = np.array(x)
            y = np.array(y)

            y = np.argmax(y, axis=1)
            x = np.argmax(one_hot_encode_id(x, n_classes=4), axis=1)

            self.confusion_matrix_plot.classes = metadata['classes']
            self.confusion_matrix_plot.add_raw_values(x, y)

        if 'roc_curves' in self.plots:
            x, y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pid', 'pid_probability',
                                                                     numpy=False, not_found_value=None, sum_multi=True)

            filter = np.argwhere(np.array([a is not None for a in y], np.bool))
            filter = filter[:, 0]

            x = [x[i] for i in filter]
            y = [y[i] for i in filter]

            x = np.array(x)
            y = np.array(y)


            self.roc_curves.classes = metadata['classes']
            x = one_hot_encode_id(x, n_classes=len(metadata['classes']))
            self.roc_curves.add_raw_values(x, y)

        if 'response_histogram' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1

            data = y[filter] / x[filter]
            data[data>3] = 3

            self.response_histogam.add_raw_values(data, tags)

        if 'response_histogram_divided' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1

            data = y[filter] / x[filter]
            data[data>3] = 3

            self.response_histogam_divided.add_raw_values(x[filter], data, tags)

        if 'response_pt_histogram' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pt', 'pt', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1

            data = y[filter] / x[filter]
            data[data>3] = 3

            self.response_pt_histogam.add_raw_values(data, tags)


        if 'response_pt_histogram_divided' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pt', 'pt', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1

            data = y[filter] / x[filter]
            data[data>3] = 3

            self.response_pt_histogam_divided.add_raw_values(x[filter], data, tags)

        if 'total_response_true_dep_to_impact' in self.plots:
            x = []
            y = []
            for g in analysed_graphs:
                truth_impact = 0
                truth_dep = 0
                pred_values = 0
                for n, att in g.nodes(data=True):
                    if att['type'] == matching_and_analysis.NODE_TYPE_TRUTH_SHOWER:
                        truth_impact += att['energy']
                        truth_dep += att['dep_energy']
                    if att['type'] == matching_and_analysis.NODE_TYPE_PRED_SHOWER:
                        pred_values += att['energy']
                x += [truth_dep/truth_impact]
                y += [pred_values/truth_impact]

            x = np.array(x)
            x[x>1.2] = 1.201
            x[x<0.8] = 0.799

            y = np.array(y)
            y[y>1.2] = 1.201
            y[y<0.8] = 0.799

            tags2 = tags.copy()
            tags2['numerator'] = '$\\sum E_{true\_dep}$'

            tags2_2 = tags.copy()
            tags2_2['numerator'] = '$\\sum E_{pred}$'

            self.total_dep_to_impact.add_raw_values(x, tags2)
            self.total_dep_to_impact.add_raw_values(y, tags2_2)

        # if 'total_response_pred_to_impact' in self.plots:
        #     x = []
        #     for g in analysed_graphs:
        #         truth_impact = 0
        #         pred_values = 0
        #         for n, att in g.nodes(data=True):
        #             if att['type'] == matching_and_analysis.NODE_TYPE_TRUTH_SHOWER:
        #                 truth_impact += att['energy']
        #             if att['type'] == matching_and_analysis.NODE_TYPE_PRED_SHOWER:
        #                 pred_values += att['energy']
        #         x += [pred_values/truth_impact]
        #
        #     x = np.array(x)
        #     x[x>1.2] = 1.201
        #     x[x<0.8] = 0.799
        #     self.total_pred_to_impact.add_raw_values(x, tags)

        if 'total_dep_pred_no_noise_to_dep_pred' in self.plots:
            x = []
            for g in analysed_graphs:
                pred_dep = 0
                pred_dep_no_noise = 0
                for n, att in g.nodes(data=True):
                    if att['type'] == matching_and_analysis.NODE_TYPE_PRED_SHOWER:
                        pred_dep_no_noise += att['dep_energy_no_noise']
                        pred_dep += att['dep_energy']
                x += [pred_dep_no_noise/(pred_dep+0.00000001)]

            x = np.array(x)
            # x[x>1.2] = 1.201
            # x[x<0.8] = 0.799
            self.total_no_noise_to_pred.add_raw_values(x, tags)

        if 'total_dep_pred_only_noise_to_dep_pred' in self.plots:
            x = []
            for g in analysed_graphs:
                pred_dep = 0
                pred_dep_only_noise = 0
                for n, att in g.nodes(data=True):
                    if att['type'] == matching_and_analysis.NODE_TYPE_PRED_SHOWER:
                        pred_dep_only_noise += att['dep_energy_only_noise']
                        pred_dep += att['dep_energy']
                x += [pred_dep_only_noise/(pred_dep+0.00000001)]

            x = np.array(x)
            # x[x>1.2] = 1.201
            # x[x<0.8] = 0.799
            self.total_only_noise_to_pred.add_raw_values(x, tags)

        if 'e_other_histogram' in self.plots:
            x,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy_others_in_vicinity', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            x[x>200]=199.9999

            self.e_other_histogram.add_raw_values(x, tags)


        if 'corr_factor_histogram' in self.plots:
            x,_ = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'corr_factor', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            # x[x>200]=199.9999
            x[x<0.7] = 0.71
            x[x>1.5] = 1.49
            self.corr_factor_histogram.add_raw_values(x, tags)


        if 'noise_assigned_to_pred_histogram' in self.plots:
            data = []
            for g in analysed_graphs:
                noise_sum_in_pred = 0.0
                for n, att in g.nodes(data=True):
                    if att['type'] == matching_and_analysis.NODE_TYPE_PRED_SHOWER:
                        noise_sum_in_pred += att['dep_energy_only_noise']
                data.append(noise_sum_in_pred)
            data = np.array(data)
            self.noise_assigned_to_pred_histogram.add_raw_values(data, tags)


        if 'noise_assigned_to_pred_to_total_noise_histogram' in self.plots:

            data = []
            for g in analysed_graphs:
                noise_sum_in_pred = 0.0
                for n, att in g.nodes(data=True):
                    if att['type'] == matching_and_analysis.NODE_TYPE_PRED_SHOWER:
                        noise_sum_in_pred += att['dep_energy_only_noise']
                data.append(noise_sum_in_pred / g.nodes['graph_data']['total_noise_rechit_sum'])

            data = np.array(data)
            self.noise_assigned_to_pred_to_total_noise_histogram.add_raw_values(data, tags)


        for plot,config in zip(self.all_plots, self.all_plots_config):
            if config['class']=='efficiency_simple':
                x, y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, config['fo'], 'energy',
                                                                         numpy=True, not_found_value=-1, sum_multi=True)
                y = np.not_equal(y, -1)
                if 'x_transform' in config:
                    x = config['x_transform'](x)

                plot.add_raw_values(x, y, tags)

            elif config['class']=='efficiency_truth_pu_adjustment':
                x, y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, config['fo'], 'energy',
                                                                         numpy=True, not_found_value=-1, sum_multi=True)
                y = np.not_equal(y, -1)
                if 'x_transform' in config:
                    x = config['x_transform'](x)

                plot.add_raw_values(x, y, tags, weights=self.truth_weights_for_pu)

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
            elif config['class']=='response_simple' or config['class']=='resolution_simple' or config['class'] == 'response_energy_spectrum_flattened'\
                    or config['class'] == 'response_truth_pu_adjustment' or config['class'] == 'resolution_truth_pu_adjustment':
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

                weights_ = None
                if config['class']=='response_energy_spectrum_flattened':
                    weights_ = weights
                elif config['class']=='response_truth_pu_adjustment' or config['class']=='resolution_truth_pu_adjustment':
                    weights_ = self.truth_weights_for_pu

                # plot.add_raw_values(x[filter], y, tags, weights=weights if config['class']=='response_energy_spectrum_flattened' else None)
                plot.add_raw_values(x[filter], y, tags, weights=weights_)
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

                if 'truth_min_cut_on' in config:
                    _, cut_on = matching_and_analysis.get_pred_matched_attribute(
                        analysed_graphs, 'energy', config['truth_min_cut_on'],
                        numpy=True, not_found_value=-1)
                    filter = np.logical_and(filter, cut_on > config['truth_min_cut_value'])

                y = ep[filter] / et[filter]

                # plt.scatter(et[filter], ep[filter], s=0.1)
                # plt.xlabel('E true')
                # plt.ylabel('E pred')
                # plt.show()

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

        if 'settings' in self.plots:
            fig = self._draw_numerics()
            pdf_others.savefig(fig)

        for plot, config in zip(self.all_plots, self.all_plots_config):
            fig = plot.draw(formatter)
            pdf_writer[config['file']].savefig(fig)

        if 'efficiency_fo_truth_pid' in self.plots:
            self.efficiency_fo_truth_pid_plot.draw(formatter)
            pdf_efficiency.savefig()

        if 'response_fo_truth_pid' in self.plots:
            self.response_fo_truth_pid_plot.draw(formatter)
            pdf_response.savefig()

        if 'confusion_matrix' in self.plots:
            fig = self.confusion_matrix_plot.draw(formatter)
            pdf_pid.savefig(fig)

            self.confusion_matrix_plot.dont_plot(3)
            fig = self.confusion_matrix_plot.draw(formatter)
            pdf_pid.savefig(fig)
            self.confusion_matrix_plot.dont_plot(None)

        if 'roc_curves' in self.plots:
            for i in range(4):
                self.roc_curves.set_primary_class(i)
                fig = self.roc_curves.draw(formatter)
                pdf_pid.savefig(figure=fig)

            for i in range(3):
                self.roc_curves.set_primary_class(i)
                self.roc_curves.dont_plot(3)
                fig = self.roc_curves.draw(formatter)
                pdf_pid.savefig(figure=fig)


        if 'response_histogram' in self.plots:
            fig = self.response_histogam.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'response_histogram_divided' in self.plots:
            fig = self.response_histogam_divided.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'response_pt_histogram' in self.plots:
            fig = self.response_pt_histogam.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'response_pt_histogram_divided' in self.plots:
            fig = self.response_pt_histogam_divided.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'total_response_true_dep_to_impact' in self.plots:
            formatter2 = lambda x: 'x=%s'%x['numerator']

            fig = self.total_dep_to_impact.draw(formatter2)
            pdf_response_histos.savefig(fig)

        # if 'total_response_pred_to_impact' in self.plots:
        #     fig = self.total_pred_to_impact.draw(formatter)
        #     pdf_response_histos.savefig(fig)

        if 'total_dep_pred_no_noise_to_dep_pred' in self.plots:
            fig = self.total_no_noise_to_pred.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'total_dep_pred_only_noise_to_dep_pred' in self.plots:
            fig = self.total_only_noise_to_pred.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'e_other_histogram' in self.plots:
            fig = self.e_other_histogram.draw(formatter)
            pdf_others.savefig(fig)

        if 'corr_factor_histogram' in self.plots:
            fig = self.corr_factor_histogram.draw(formatter)
            pdf_others.savefig(fig)

        if 'noise_assigned_to_pred_histogram' in self.plots:
            fig = self.noise_assigned_to_pred_histogram.draw(formatter)
            pdf_others.savefig(fig)

        if 'noise_assigned_to_pred_to_total_noise_histogram' in self.plots:
            fig = self.noise_assigned_to_pred_to_total_noise_histogram.draw(formatter)
            pdf_others.savefig(fig)


        pdf_efficiency.close()
        pdf_response.close()
        pdf_fake_rate.close()
        pdf_others.close()
        pdf_pid.close()
        pdf_resolution.close()
        pdf_response_histos.close()

        plt.close('all')
