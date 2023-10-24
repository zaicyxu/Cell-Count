import numpy as np
SAVE_FOCUS_IM = False
IM_W = 3072
IM_H = 2048
SCALE = min(1920 * 0.8 / IM_W, 1080 * 0.8 / IM_H)
VIS_W = int(IM_W * SCALE)
VIS_H = int(IM_H * SCALE)
IM_NUM = 3
SAVE_ROOT = '/home/ailab/python_sdkcam/samples/'
DISK_TYPE = True

POOL_STEPS = {
	'1': [1869000, 1914500, 1962000],
	'2': [1485000, 1541500, 1568000],
	'3': [1120000, 1168500, 1200000],
	'4': [748000, 795500, 843000],
	'5': [377000, 422500, 470000],
	'6': [4000, 49500, 100000]
}
IM_BUFFER = np.zeros((IM_NUM, IM_H, IM_W), np.uint8)
VIS_BUFFER_ORG = np.zeros((IM_NUM, IM_H, IM_W, 3), np.uint8)
