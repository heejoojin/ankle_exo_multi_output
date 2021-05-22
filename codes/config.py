import os

SPEED_LIST = ['speed0', 'speed2', 'speed3']
ACT_LIST = ['act0', 'act1', 'act2', 'act3']

TEST_TYPE_LIST = SPEED_LIST + ACT_LIST
TRAIN_TYPE_LIST = ['all', 'left', 'right']

ANGLE_COLOR = ['orangered', 'darksalmon', 'lightsalmon', 'peachpuff'] # ['dodgerblue', 'deepskyblue', 'powderblue', 'paleturquoise']
TORQUE_COLOR = ['orangered', 'darksalmon', 'lightsalmon', 'peachpuff']

COLORS = ['dodgerblue', 'deepskyblue', 'powderblue', 'orangered', 'darksalmon', 'lightsalmon', 'peachpuff']

ORIGINAL_DATA_PATH = os.path.join(os.pardir, 'original_data')
DATA_PATH = os.path.join(os.pardir, 'data')
RESULT_PATH = os.path.join(os.pardir, 'results')
PLOT_PATH = os.path.join(os.pardir, 'plots')
MODEL_TYPE = ['100', 'best']

LABEL_DICT = {'speed0': '0.8 m/s', 'speed2': '1.0 m/s', 'speed3': '1.2 m/s', 'act0': '0 Nm', 'act1': '10 Nm', 'act2': '20 Nm', 'act3': '30 Nm'}

MODEL_LIST = ['left', 'right']

PARAMS = {'legend.fontsize': 20,
        'axes.labelsize': 30,
        'axes.titlesize': 40,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'lines.linewidth': 10}
         
K_FOLD = 5