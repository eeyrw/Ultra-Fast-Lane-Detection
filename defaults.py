from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 1
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4


_C.DATASET = CN()
_C.DATASET.NAME='Tusimple'
_C.DATASET.ROOT = 'E:/Tusimple'
_C.DATASET.RAW_IMG_SIZE = [1280, 720] #h,w
_C.DATASET.TRAIN_PROPORTION=0.1
_C.DATASET.TEST_PROPORTION=1
_C.DATASET.NUM_LANES = 4


_C.TRAIN = CN()
_C.TRAIN.EPOCH = 1
_C.TRAIN.BATCH_SIZE = 28
_C.TRAIN.OPTIMIZER = 'Adam'  #['SGD','Adam']
_C.TRAIN.LR = 1e-3
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.SCHEDULER = 'multi'  #['multi', 'cos']
_C.TRAIN.STEPS = [10,20]
_C.TRAIN.GAMMA = 0.1
_C.TRAIN.WARMUP = 'linear'
_C.TRAIN.WARMUP_ITERS = 500

# NETWORK
_C.NETWORK = CN()
_C.NETWORK.BACKBONE = '18'
_C.NETWORK.GRIDING_NUM = 100
_C.NETWORK.USE_AUX = True
_C.NETWORK.USE_SPP = False

# LOSS
_C.LOSS = CN()
_C.LOSS.SIM_LOSS_W = 1.0
_C.LOSS.SHP_LOSS_W = 0.0

# EXP
_C.EXP = CN()
_C.EXP.NOTE = ''
_C.EXP.LOG_PATH = '../log'
# FINETUNE or RESUME MODEL PATH
_C.EXP.FINETUNE = None #r"tusimple_b32_ep098.pth"
_C.EXP.RESUME = None

# TEST
_C.TEST = CN()
_C.TEST.MODEL = r'..\log\tusimple_b32_ep098.pth'
_C.TEST.WORK_DIR = '../test_tmp'
_C.TEST.INTERVAL = 3
_C.TEST.DURING_TRAIN = True



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`