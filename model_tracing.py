import sys
sys.path.append('./NWPU-Crowd-Sample-Code/')

import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from misc import layer
from models import counters

import pdb

import argparse

parser = argparse.ArgumentParser(description='Trace Pytorch Model.')
parser.add_argument('--model-name', type=str, default='CSRNet',
                    help='model name(should be in models.counters)')
parser.add_argument('--input-shape', type=str,
                    help='image shape. e.t (1,3,224,224)')
parser.add_argument('--weights-path', type=str,
                    help='weights path')
parser.add_argument('--out-path', type=str,
                    help='traced model path')

args = parser.parse_args()

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()    
        ccnet =  getattr(getattr(counters, model_name), model_name)

        gs_layer = getattr(layer, 'Gaussianlayer')

        self.CCN = ccnet()
        self.gs = gs_layer()
       
        self.loss_mse_fn = nn.MSELoss()

        
    @property
    def loss(self):
        return self.loss_mse_fn
    

    def forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map


net = CrowdCounter([], 'CSRNet')
net.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
net.eval()

args.input_shape = list(map(int, args.input_shape[1:-1].split(',')))
traced_model = torch.jit.trace(net, torch.rand(args.input_shape))
traced_model.save(args.out_path)
