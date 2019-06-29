import os
import sys
import logging
from time import strftime, localtime, time
import torch
from torch import nn, optim
from torch.backends import cudnn
import numpy as np
from dot_util import DotDict

CONFIG = {
    # Choose dataset
    'dataset': 'laptop',  # Other choices: ['restaurant', 'laptop']
    'train_file': 'data/laptop/Laptops_Train.xml.seg',
    'test_file': 'data/laptop/Laptops_Test_Gold.xml.seg',
    # Choose model
    'model': 'bert_aen',  # Other choice: ['bert_spc']
    'log_file': 'logs/{}-{}-{}.log'.format('bert_aen', 'twitter',
                                           strftime("%y%m%d-%H%M",
                                                    localtime())),
    # Pretrained model parameters
    'bert_vocab_path': 'bert/vocab.txt',
    'bert_model_path': 'bert/uncased.tar.gz',
    'bert_dim': 768,
    'hidden_dim': 300,
    'polarities_dim': 3,
    # Parameters of model
    'dropout_rate': 0.1,
    'num_epoch': 5,
    'batch_size': 64,
    'valid_ratio': 0.3,
    'max_seq_len': 80,
    'log_step': 5,
    'cross_val_fold': 10,
    # Initializer and optimizer
    'initializer': nn.init.xavier_normal_,
    'optimizer': optim.Adam,
    'learning_rate': 2e-5,
    'ridge_reg': 0.01,
    # Device parameters
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

# Initialize directories
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('state_dict'):
    os.mkdir('state_dict')

# Transform into a dict with dot access
CONFIG = DotDict(CONFIG)

# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler(CONFIG.log_file))

# Random seed
seed = int(time())
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False
