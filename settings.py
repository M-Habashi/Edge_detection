import os
import torch


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

