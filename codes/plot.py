import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config as c
import metric
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 30

def plot_grouped_rmse(width=20, height=20):

    bar_width = 0.35
    fig, (speed, act) = plt.subplots(figsize=(width, height), nrows=2)
    
    for test_type, _list in zip(['speed', 'act'], [c.SPEED_LIST, c.ACT_LIST]):

        if test_type == 'speed':
            ax = speed
            labels = ['0.8', '1.0', '1.2']
            ax.set_title('Leave-One-Speed-Out', fontweight='semibold')
            ax.set_xlabel('Walking Speed [m/s]', fontweight='semibold')

        elif test_type == 'act':
            ax = act
            labels = ['0', '10', '20', '30']
            ax.set_title('Leave-One-Torque-Out', fontweight='semibold')
            ax.set_xlabel('Torque [Nm]', fontweight='semibold')
            
        x_pos = np.arange(len(labels))
        
        for name in c.MODEL_LIST:
            rmse = []
            std = []

            for i, file_name in enumerate(_list):
                
                # gait_percentage_df = pd.DataFrame()
                file_name = '%s_%s'%(file_name, name)

                # for k in range(1, c.K_FOLD + 1):
                
                    # data = pd.read_csv(os.path.join(os.path.join(c.RESULT_PATH, file_name), 'raw_result_r_%d.csv'%k))
                    # gait_percentage_df = pd.concat([gait_percentage_df, data], axis=0, ignore_index=True)
                data_rmse = pd.read_csv(os.path.join(os.path.join(c.RESULT_PATH, file_name), 'rmse.csv')).to_numpy()
                
                std.append(np.std(data_rmse))
                rmse.append(np.mean(data_rmse))
                
                # _rmse = metric.gait_phase_rmse(torch.Tensor(gait_percentage_df[['gt_x', 'gt_y']].to_numpy()), torch.Tensor(gait_percentage_df[['pred_x', 'pred_y']].to_numpy()))
                # rmse.append(_rmse.item())
                # std.append(_std.item())

            if 'left' in name:
                rects = ax.bar(x_pos - bar_width/2, rmse, bar_width, yerr=std, ecolor='grey', label='Left Ankle', color='powderblue')
            elif 'right' in name:
                rects = ax.bar(x_pos + bar_width/2, rmse, bar_width, yerr=std, ecolor='grey', label='Right Ankle', color='darksalmon')
            for j, rect in enumerate(rects):
                height = rect.get_height()
                ax.annotate('%.2f'%height,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, std[j]*100+30),
                            textcoords='offset points',
                            ha='center', va='bottom')
        
        ax.set_ylabel('Gait Phase Estimated RMSE [%]', fontweight='semibold')
        ax.set_ylim([0, 4])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(c.PLOT_PATH, 'grouped_rmse.png'))
    plt.savefig(os.path.join(c.PLOT_PATH, 'grouped_rmse.svg'))
    plt.close(fig)

def plot_raw_data(width=20, height=20):

    fig = plt.figure(figsize=(width, height))
    
    speed_angle = fig.add_subplot(221)
    act_angle = fig.add_subplot(222)
    speed_torque = fig.add_subplot(223)
    act_torque = fig.add_subplot(224)

    for i, test_type in enumerate(c.TEST_TYPE_LIST):

        test_data_path = os.path.join(os.path.join(c.DATA_PATH, 'all'), test_type)
        data_list = os.listdir(test_data_path)

        angle_df = pd.DataFrame()
        torque_df = pd.DataFrame()

        for file_name in data_list:
            
            file_data_path = os.path.join(test_data_path, file_name)
            columns = ['ankle_angle', 'ankle_torque_from_current', 'gait_phase']
            data = pd.read_csv(file_data_path, usecols=columns).dropna().reset_index(drop=True)

            start_idx = np.where(data['gait_phase'].to_numpy() == 0.0)[0].tolist()[0]
            end_idx = np.where(data['gait_phase'].to_numpy() == 0.0)[0].tolist()[-1]
            data['gait_phase'] = data['gait_phase'].round(2) * 100
            angle_df = pd.concat([angle_df, data.loc[start_idx:end_idx, ['gait_phase', 'ankle_angle']]], axis=0, ignore_index=True)
            torque_df = pd.concat([torque_df, data.loc[start_idx:end_idx, ['gait_phase', 'ankle_torque_from_current']]], axis=0, ignore_index=True)
        
        angle = angle_df.groupby(['gait_phase'], as_index=False).agg({'ankle_angle':'mean'}).to_numpy()
        torque = torque_df.groupby(['gait_phase'], as_index=False).agg({'ankle_torque_from_current': 'mean'}).to_numpy()

        label = c.LABEL_DICT[test_type]
        if 'speed' in test_type:
            speed_angle.plot(angle[:,0], angle[:,1], label=label, color=c.ANGLE_COLOR[i], zorder=i+1, linewidth=c.LINEWIDTH)
            speed_torque.plot(torque[:,0], torque[:,1], label=label, color=c.TORQUE_COLOR[i], zorder=i+1, linewidth=c.LINEWIDTH)
            
        elif 'act' in test_type:
            act_angle.plot(angle[:,0], angle[:,1], label=label, color=c.ANGLE_COLOR[i - len(c.SPEED_LIST)], zorder=i+1, linewidth=c.LINEWIDTH)
            act_torque.plot(torque[:,0], torque[:,1], label=label, color=c.TORQUE_COLOR[i - + len(c.SPEED_LIST)], zorder=i+1, linewidth=c.LINEWIDTH)

    for subplot in [speed_angle, act_angle]:
        subplot.set_title('Ankle Angle', fontweight='semibold')
        subplot.set_xlabel('Gait Phase [%]', fontweight='semibold')
        subplot.set_ylabel('Ankle Angle [°]', fontweight='semibold')
        subplot.set_xlim([0, 100])
        subplot.set_ylim([-10, 20])
        plt.yticks(np.arange(-10, 21, 5))
        subplot.axhline(0, color='lightgrey', linestyle='dashed', zorder=0, linewidth=c.LINEWIDTH)
        subplot.legend(loc='upper right')
        
    for subplot in [speed_torque, act_torque]:
        subplot.set_title('Ankle Moment', fontweight='semibold')
        subplot.set_xlabel('Gait Phase [%]', fontweight='semibold')
        subplot.set_ylabel('Ankle Moment [Nm/kg]', fontweight='semibold')
        subplot.set_xlim([0, 100])
        subplot.set_ylim([-10, 30])
        plt.yticks(np.arange(-10, 31, 5))
        subplot.axhline(0, color='lightgrey', linestyle='dashed', zorder=0, linewidth=c.LINEWIDTH)
        subplot.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(c.PLOT_PATH, 'raw_data.png'))
    plt.savefig(os.path.join(c.PLOT_PATH, 'raw_data.svg'))
    plt.close(fig)

def plot_all_edges(width=20, height=20, start_time=-0.05, end_time=0.05):

    # classification result
    time_labels = []
    time_options = 0

    while start_time <= end_time:
        
        _str = str(start_time)
        time_labels.append(_str)
        time_options += 1
        start_time = start_time + 0.005
        start_time = round(start_time, 4)
        
    time_options = int((time_options - 1) / 2)
    x_pos = np.arange(time_options * 2 + 1)

    for name in c.MODEL_LIST:
        fig = plt.figure(figsize=(width, height))
        speed_rising = fig.add_subplot(221)
        speed_falling = fig.add_subplot(222)
        act_rising = fig.add_subplot(223)
        act_falling = fig.add_subplot(224)

        for i, file_name in enumerate(c.TEST_TYPE_LIST):
           
            file_name = '%s_%s'%(file_name, name)
            heel_strike_toe_off_path = os.path.join(os.path.join(os.path.join(c.RESULT_PATH, file_name), 'npy_data'), 'heel_strike_toe_off.npy')
            heel_strike_toe_off = np.load(heel_strike_toe_off_path)

            rising_edge = np.zeros((time_options * 2 + 1, ))
            falling_edge = np.zeros((time_options * 2 + 1, ))
            heel_count = 0.0
            toe_count = 0.0

            label = c.LABEL_DICT[file_name.split('_')[0]] # [:-1], c.LABEL_DICT[file_name.split('_')[0][-1])

            for j in range(1, c.K_FOLD + 1):
                _result_path = os.path.join(os.path.join(c.RESULT_PATH, file_name), 'result_c_%d.csv'%j)
                result = pd.read_csv(_result_path).to_numpy()

                heel_strike = heel_strike_toe_off[:, 0]
                heel_strike_idx = np.where(heel_strike == 1)[0].tolist()

                toe_off = heel_strike_toe_off[:, 1]
                toe_off_idx = np.where(toe_off == 1)[0].tolist()
                
                for k in heel_strike_idx:
                    
                    start_idx = k - time_options
                    end_idx = k + time_options + 1

                    if start_idx < 0 or end_idx >= len(result):
                        continue
                    diff = np.abs(result[start_idx : end_idx, 0] - result[start_idx : end_idx, 1])
                    rising_edge = rising_edge + diff
                    heel_count += (end_idx - start_idx)

                for k in toe_off_idx:
                    start_idx = k - time_options
                    end_idx = k + time_options + 1
                    if start_idx < 0 or end_idx >= len(result):
                        continue
                    diff = np.abs(result[start_idx : end_idx, 0] - result[start_idx : end_idx, 1])
                    falling_edge = falling_edge + diff
                    toe_count += (end_idx - start_idx)
            
            rising_edge = rising_edge / c.K_FOLD / heel_count * 100
            falling_edge = falling_edge / c.K_FOLD / toe_count * 100
            
            if 'speed' in file_name:
                speed_rising.plot(rising_edge, label=label, color=c.COLORS[i], linewidth=c.LINEWIDTH)
                speed_falling.plot(falling_edge, label=label, color=c.COLORS[i], linewidth=c.LINEWIDTH)
                speed_rising.annotate('%.2f'%np.amax(rising_edge), (np.argmax(rising_edge), np.amax(rising_edge)), ha='center')
                speed_falling.annotate('%.2f'%np.amax(falling_edge), (np.argmax(falling_edge), np.amax(falling_edge)), ha='center')

            elif 'act' in file_name:
                act_rising.plot(rising_edge, label=label, color=c.COLORS[i], linewidth=c.LINEWIDTH)
                act_falling.plot(falling_edge, label=label, color=c.COLORS[i], linewidth=c.LINEWIDTH)
                act_rising.annotate('%.2f'%np.amax(rising_edge), (np.argmax(rising_edge), np.amax(rising_edge)), ha='center')
                act_falling.annotate('%.2f'%np.amax(falling_edge), (np.argmax(falling_edge), np.amax(falling_edge)), ha='center')

        speed_rising.set_title('Speed - Swing to Stance', fontweight='semibold')
        speed_falling.set_title('Speed - Stance to Swing', fontweight='semibold')
        act_rising.set_title('Torque - Swing to Stance', fontweight='semibold')
        act_falling.set_title('Torque - Stance to Swing', fontweight='semibold')

        for subplot in [speed_rising, speed_falling, act_rising, act_falling]:
            subplot.set_xlabel('Time Difference [Seconds]', fontweight='semibold')
            subplot.set_ylabel('Error Occurance [%]', fontweight='semibold')
            subplot.set_xticks(x_pos)
            subplot.set_xticklabels(time_labels)
            
            for i, label in enumerate(subplot.xaxis.get_ticklabels()):
                if float(label.get_text()) % 0.025 != 0:
                    label.set_visible(False)
                if float(label.get_text()) == 0.0:
                    subplot.axvline(i, color='lightgrey', linestyle='dashed', linewidth=c.LINEWIDTH, zorder=0)

            subplot.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(c.PLOT_PATH, 'edges_%s.png'%name))
        plt.savefig(os.path.join(c.PLOT_PATH, 'edges_%s.svg'%name))
        plt.close(fig)
