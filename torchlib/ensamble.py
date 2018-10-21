
# Ensamble method
# autor: Pedro Diamel Marrero Fernandez
#
#            + --------+
#            | PoolNet |                 ruler
#            |---------+                   |
#   +--+     | model_1 | ------------+--->-+
# --|  | --> | model_2 | -----+------c--->-+ 
#   +--+     |   ...   |      |      |     +---> final desition
#    |       |         |      |      |     |     ruler
#    |       | model_n | ---+-c------c--->-+ 
#    |       +---------+    | |      |
#    |                      | |      |
#    |       +---------+    | |      | 
#    +-------| Expert  |----+-+- ... +
#            +---------+ 
#

import torch
import torch.nn as nn


class PoolNet( nn.Module ):
    r"""PoolNet for ensamble methods
    Execute various training models
    Args:        
    """

    def __init__(self):
        super(PoolNet, self).__init__()
        nets = []

    def add( model ):
        nets.append( model )

    def forward(self, x):
        ys = [] # [n, dim, m]
        for net in nets:
            z = net(x)
            ys.append(z)
        ys = torch.concatenate( ys, dim=2 )
        return ys
    

