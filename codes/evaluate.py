import os
import csv
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config as c
import metric
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc
# from metric import *

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 40

class Test:
    def __init__(self, **kwargs):
        
        self.device = kwargs['device']
        self.k_fold = kwargs['k_fold']
        self.task = kwargs['task']
        self.data_loader = kwargs['data_loader']
        self.data_type = kwargs['data_type']

        self.result_path = kwargs['result_path']
        self.heel_strike_toe_off_path = os.path.join(os.path.join(self.result_path, 'npy_data'), 'heel_strike_toe_off.npy')
        self.roc_graph_path = os.path.join(self.result_path, 'roc.png')
        self.edges_path = os.path.join(self.result_path, 'edges.png')

        self.rmse_path = os.path.join(self.result_path, 'rmse.csv')
        self.cm_path = os.path.join(self.result_path, 'classification_metrics.csv')
        
        for _path in [self.rmse_path, self.cm_path]:
            if os.path.isfile(_path):
                os.remove(_path)
        for _mode in ['x_train', 'x_val', 'y_train', 'y_val']:
            _path = os.path.join(os.path.join(self.result_path, 'npy_data'), '%s.npy'%_mode)
            if os.path.isfile(_path):
                os.remove(_path)

        self.colors = ['lightcoral', 'firebrick', 'chocolate', 'brown', 'orange', 'olive', 'green', 'teal', 'midnightblue', 'purple']
        self.roc_fig = plt.figure(figsize=(20, 20))
        self.roc_ax = self.roc_fig.add_subplot(111)
        self.roc_ax.set_title('Receiver Operating Characteristic', fontweight='semibold')

        start_time = -0.1
        end_time = 0.1
        self.time_labels = []
        while start_time <= end_time:
            _str = str(start_time)
            self.time_labels.append(_str)
            start_time = start_time + 0.005
            start_time = round(start_time, 4)
        self.time_options = 20
        self.swing_to_stance = np.zeros((2 * self.time_options + 1, ))
        self.stance_to_swing = np.zeros((2 * self.time_options + 1, ))

        self._range = 1500

    def test_classification(self, output, target):
        
        _roc_result = np.concatenate([target.numpy(), output.numpy()], axis=1)
        _output = torch.sigmoid(output)
        _output = (_output > 0.5).float()
        _result = np.concatenate([target.numpy(), _output.numpy()], axis=1)
        _metrics = metric.classification_metrics(_output, target)
        
        return _roc_result, _result, _metrics
        
    def test_regression(self, output, target): # , mask=None):
        
        _gait_percentage = metric.polar_coordiantes_to_gait_percentage(output=output, target=target)
        # _gait_percentage[:, 0] = target
        # _gait_percentage[:, 1] = output

        return _gait_percentage
    
    def do(self):
        
        for fold_idx in range(1, self.k_fold + 1):

            model_path = os.path.join(self.result_path, 'model_%d_100.pt'%fold_idx)
            csv_result_path = os.path.join(self.result_path, 'result_%d.csv'%fold_idx)

            csv_result_r_path = os.path.join(self.result_path, 'result_r_%d.csv'%fold_idx)
            csv_result_c_path = os.path.join(self.result_path, 'result_c_%d.csv'%fold_idx)

            raw_result_path = os.path.join(self.result_path, 'raw_result_r_%d.csv'%fold_idx)

            model = torch.load(model_path, map_location='cpu')
            
            model.to(self.device)

            model.eval()

            result = np.empty((0, 2))
            result_r = np.empty((0, 2))
            result_c = np.empty((0, 2))
            raw_result = np.empty((0, 4))
            roc_result = np.empty((0, 2))

            metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'npv': 0.0, 'f1score': 0.0}
            start_time = time.time()

            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(self.data_loader):
                    x = x.to(self.device)
                    output = model(x)
                    
                    if self.task == 'regression':
                        output = output.cpu()
                        _gait_percentage = self.test_regression(output, y)
                        result = np.concatenate([result, _gait_percentage.numpy()], axis=0)
                        _raw_result = np.concatenate([y, output], axis=1)
                        raw_result = np.concatenate([raw_result, _raw_result], axis=0)
                        
                    elif self.task == 'classification':
                        output = output.cpu()
                        _roc_result, _result, _metrics = self.test_classification(output, y)
                        roc_result = np.concatenate([roc_result, _roc_result], axis=0)
                        result = np.concatenate([result, _result], axis=0)
                        for key in metrics.keys():
                            metrics[key] += _metrics[key]

                    elif self.task == 'multi':
                        out_r, out_c = output
                        out_r = out_r.cpu()
                        out_c = out_c.cpu()

                        #############################################################
                        # getting gait phase estimation, rmse, standard deviation after gait phase conversion from its polar coordinates
                        _gait_percentage = self.test_regression(out_r, y[:, :-1]) #, mask=y[:, -1])
                        result_r = np.concatenate([result_r, _gait_percentage.numpy()], axis=0)
                        #############################################################
                        
                        #############################################################
                        # saving raw test result (= x,y polar coordinates) before gait conversion
                        # indices = torch.nonzero(y[:, -1], as_tuple=True)
                        # indices = indices[0]
                        indices = torch.nonzero(_gait_percentage[:,0] <= 0.60, as_tuple=True)[0]
                        # print(len(indices))
                        _target = y[indices, :-1]
                        _output = out_r[indices, :]
                        _raw_result = np.concatenate([_target, _output], axis=1)
                        raw_result = np.concatenate([raw_result, _raw_result], axis=0)
                        #############################################################
                        
                        #############################################################
                        # saving roc curve after gait phase classification
                        _roc_result, _result, _metrics = self.test_classification(out_c, y[:, -1].unsqueeze(dim=1))
                        roc_result = np.concatenate([roc_result, _roc_result], axis=0)
                        result_c = np.concatenate([result_c, _result], axis=0)
                        for key in metrics.keys():
                            metrics[key] += _metrics[key]
                        #############################################################
            
            rmse = metric.gait_phase_rmse(target=torch.Tensor(raw_result[:, :2]), output=torch.Tensor(raw_result[:, 2:]))
            rmse = rmse.item()
            print(rmse)
            # std = std.item()

            end_time = time.time() - start_time

            #############################################################
            # logging & plotting test results
            if self.task == 'multi':
                result_r_df = pd.DataFrame(result_r, columns=['ground_truth', 'prediction'])
                result_c_df = pd.DataFrame(result_c, columns=['ground_truth', 'prediction'])
                result_r_df.to_csv(csv_result_r_path, index=False)
                result_c_df.to_csv(csv_result_c_path, index=False)
                raw_result_df = pd.DataFrame(raw_result, columns=['gt_x', 'gt_y', 'pred_x', 'pred_y'])
                raw_result_df.to_csv(raw_result_path, index=False)

            else:
                result_df = pd.DataFrame(result, columns=['ground_truth', 'prediction'])
                result_df.to_csv(csv_result_path, index=False)
            
            if self.task == 'regression' or self.task == 'multi':
                
                print('RMSE: %1.5f | Elapsed Time: %0f'%(rmse, end_time))

                plot_path_png = os.path.join(self.result_path, 'gait_phase_regression_%d.png'%fold_idx)
                plot_path_svg = os.path.join(self.result_path, 'gait_phase_regression_%d.svg'%fold_idx)

                with open(self.rmse_path, 'a', newline='') as f:
                    csv_writer = csv.DictWriter(f, fieldnames=['rmse'])
                    csv_writer.writerow({'rmse': rmse})

                if self.task == 'multi':
                    
                    #############################################################
                    fig = plt.figure(figsize=(20, 8))
                    dotted = result_r_df['ground_truth'].copy()

                    result_r_df['ground_truth'] = result_r_df['ground_truth'].where(result_r_df['ground_truth'] <= 0.60, np.nan)
                    result_r_df['prediction'] = result_r_df['prediction'].where(result_r_df['ground_truth'] <= 0.60, np.nan)
                    
                    plt.axhline(60.0, color='lightgray', linestyle='dashed', zorder=0)

                    plt.plot(dotted.iloc[:self._range] * 100, color='lightgrey', linestyle='dashed')
                    plt.plot(result_r_df.iloc[:self._range, 0] * 100, label='Groud Truth', color='powderblue')
                    plt.plot(result_r_df.iloc[:self._range, 1] * 100, label='Prediction', color='darksalmon')
                    if self.data_type == 'left':
                        plt.title('Left Ankle', fontweight='semibold')
                    elif self.data_type == 'right':
                        plt.title('Right Ankle', fontweight='semibold')

                    plt.xticks([])
                    plt.ylabel('Estimated Gait Phase [%]', fontweight='semibold')
                    plt.xlabel('Time [Seconds]', fontweight='semibold')
                    plt.legend(loc='upper right')
                    plt.tight_layout()
                    plt.savefig(plot_path_png)
                    plt.savefig(plot_path_svg)
                    plt.close(fig)
                    #############################################################
                    
                else:
                    self.plot_regression(result, plot_path_png, plot_path_svg, rmse)
                
            if self.task == 'classification' or self.task == 'multi':
                for key in metrics.keys():
                    metrics[key] = metrics[key] / len(self.data_loader)
                print('Classification result')
                print(metrics)
                print('Elapsed Time: %0f'%(end_time))
                plot_path_png = os.path.join(self.result_path, 'classification_%d.png'%fold_idx)
                plot_path_svg = os.path.join(self.result_path, 'classification_%d.svg'%fold_idx)
                with open(self.cm_path, 'a', newline='') as f:
                    csv_writer = csv.DictWriter(f, fieldnames=['accuracy', 'precision', 'recall', 'specificity', 'npv', 'f1score'])
                    csv_writer.writerow(metrics)
                
                if self.task == 'multi':
                    self.set_edges(result_c)
                    self.plot_classification(result_c, plot_path_png, plot_path_svg)
                else:
                    self.set_edges(result)
                    self.plot_classification(result, plot_path_png, plot_path_svg)
                self.set_roc_curve(fold_idx, roc_result)
                
        if self.task == 'classification' or self.task == 'multi':
            self.plot_roc_curve()
            self.plot_edges()
        #############################################################
    
    def set_roc_curve(self, fold_idx, roc_result):
        fpr, tpr, thresholds = roc_curve(roc_result[:,0], roc_result[:,1], pos_label=1)
        auroc = auc(fpr, tpr)
        self.roc_ax.plot(fpr, tpr, 'b', color=self.colors[fold_idx-1], label='ROC fold %d (AUC = %0.2f)'%(fold_idx, auroc))

    def plot_roc_curve(self):
        x_min, x_max = 0, 0.3
        y_min, y_max = 0.7, 1
        # self.roc_ax.plot([0, 1], [0, 1], 'k--', )
        self.roc_ax.set_xlim([x_min, x_max])
        self.roc_ax.set_ylim([y_min, y_max])
        self.roc_ax.set_ylabel('True Positive Rate', fontweight='semibold')
        self.roc_ax.set_xlabel('False Positive Rate', fontweight='semibold')
        self.roc_ax.legend(loc='lower right')
        self.roc_fig.savefig(self.roc_graph_path)
        plt.close(self.roc_fig)
    
    def plot_regression(self, result: np.ndarray, plot_path_png, plot_path_svg):
        
        fig = plt.figure(figsize=(20, 8))
        
        ground_truth = result[:self._range, 0] * 100
        prediction = result[:self._range, 1] * 100
        plt.plot(ground_truth, label='Groud Truth', color='powderblue')
        plt.plot(prediction, label='Prediction', color='darksalmon')
        if self.data_type == 'left':
            plt.title('Left Ankle', fontweight='semibold')
        elif self.data_type == 'right':
            plt.title('Right Ankle', fontweight='semibold')

        plt.xticks([])
        plt.ylabel('Estimated Gait Phase [%]', fontweight='semibold')
        plt.xlabel('Time [Seconds]', fontweight='semibold')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(plot_path_png)
        plt.savefig(plot_path_svg)
        plt.close(fig)

    def plot_classification(self, result: np.ndarray, plot_path_png, plot_path_svg):
        fig = plt.figure(figsize=(20, 8))
        plt.plot(result[:self._range, 0], label='Groud Truth', color='powderblue')
        plt.plot(result[:self._range, 1], label='Prediction', color='darksalmon')
        if self.data_type == 'left':
            plt.title('Left Ankle', fontweight='semibold')
        elif self.data_type == 'right':
            plt.title('Right Ankle', fontweight='semibold')
        plt.xticks([])
        plt.ylabel('Swing: 0 | Stance: 1', fontweight='semibold')
        plt.xlabel('Time [Seconds]', fontweight='semibold')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(plot_path_png)
        plt.savefig(plot_path_svg)
        plt.close(fig)
    
    def set_edges(self, result: np.ndarray):
        
        heel_strike_toe_off = np.load(self.heel_strike_toe_off_path)
    
        heel_strike = heel_strike_toe_off[:, 0]
        heel_strike_idx = np.where(heel_strike == 1)[0].tolist()

        toe_off = heel_strike_toe_off[:, 1]
        toe_off_idx = np.where(toe_off == 1)[0].tolist()
        
        for i in heel_strike_idx:
            start_idx = i - self.time_options
            end_idx = i + self.time_options + 1
            if start_idx >= 0 and end_idx < result.shape[0]:
                diff = np.abs(result[start_idx : end_idx, 0] - result[start_idx : end_idx, 1])
                self.swing_to_stance = self.swing_to_stance + diff

        for i in toe_off_idx:
            start_idx = i - self.time_options
            end_idx = i + self.time_options + 1
            if start_idx >= 0 and end_idx < result.shape[0]:
                diff = np.abs(result[start_idx : end_idx, 0] - result[start_idx : end_idx, 1])
                self.stance_to_swing = self.stance_to_swing + diff
    
    def plot_edges(self):

        self.swing_to_stance = self.swing_to_stance / self.k_fold
        self.stance_to_swing = self.stance_to_swing / self.k_fold
        
        self.swing_to_stance = self.swing_to_stance.tolist()
        self.stance_to_swing = self.stance_to_swing.tolist()

        x_pos = np.arange(len(self.time_labels))
        width = 0.35 
        fig = plt.figure(figsize=(30, 8))
        ax = fig.add_subplot(111)

        ax.bar(x_pos - width/2, self.swing_to_stance, width, label='Stance to Swing | Falling Edge', color='powderblue')
        ax.bar(x_pos + width/2, self.stance_to_swing, width, label='Swing to Stance | Rising Edge', color='darksalmon')
    
        ax.set_title('Classification Errors', fontweight='semibold')
        ax.set_xlabel('Time Difference [Seconds]', fontweight='semibold')
        ax.set_ylabel('Error Occurance', fontweight='semibold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.time_labels, size=12)
        ax.legend()

        plt.tight_layout()
        fig.savefig(self.edges_path)
        plt.close(fig)

