import numpy as np
import torch

class EngineProvide(object):
    r""" Engine generate dataset provide
    Args:
        pathname: path of dataset
    """
    def __init__(self, pathname ):
        self.pathname = pathname
        self.data = torch.load( pathname )
        self.labels = self.data['Y']

    def __len__(self):
        return len(self.data['Y'])


    def __getitem__(self, idx):
        z, y, p = self.data['Z'][idx], self.data['Y'][idx], self.data['P'][idx]
        return z, y, p
