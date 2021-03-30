import os

SPEED_LIST = ['speed0', 'speed2', 'speed3']
ACT_LIST = ['act0', 'act1', 'act2', 'act3']

TEST_TYPE_LIST = SPEED_LIST + ACT_LIST
TRAIN_TYPE_LIST = ['all', 'left', 'right']

ANGLE_COLOR = ['dodgerblue', 'deepskyblue', 'powderblue', 'paleturquoise']
TORQUE_COLOR = ['orangered', 'darksalmon', 'lightsalmon', 'peachpuff']

COLORS = ['dodgerblue', 'deepskyblue', 'powderblue', 'orangered', 'darksalmon', 'lightsalmon', 'peachpuff']

ORIGINAL_DATA_PATH = os.path.join(os.pardir, 'original_data')
DATA_PATH = os.path.join(os.pardir, 'data')
RESULT_PATH = os.path.join(os.pardir, 'results')
PLOT_PATH = os.path.join(os.pardir, 'plots')
MODEL_TYPE = ['100', 'best']

MODEL_LIST = ['left', 'right']
LINEWIDTH = 5
K_FOLD = 5