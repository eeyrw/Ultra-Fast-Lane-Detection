# DATA
dataset='Tusimple'
data_root = None
raw_img_size = [1280, 720] #h,w
train_ds_proportion=1
test_ds_proportion=1

# TRAIN
epoch = 70
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi'     #['multi', 'cos']
steps = [20,40] #epoch
gamma  = 0.1
warmup = 'linear'
warmup_iters = 500 #sample iteration

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = None

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 4
test_interval = 1
test_during_train = True