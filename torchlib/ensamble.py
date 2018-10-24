
# Ensamble method
# autor: Pedro Diamel Marrero Fernandez
#
#            + --------+
#            | PoolNet |                  
#            |---------+                   
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
import torch.nn.functional as F


class PoolNet( nn.Module ):
    r"""PoolNet for ensamble methods
    Execute various training models
    Args:        
    """

    def __init__(self):
        super(PoolNet, self).__init__()
        self.nets = []

    def __len__(self):
        return len( self.nets )

    def add(self, model ):
        self.nets.append( model )

    def forward(self, x):
        ys = [] # [n, dim, m]
        for net in self.nets:
            z = net(x)
            ys.append( z )
        ys = torch.stack( ys, dim=2 )
        return ys


class NeuralNetEnsamble( object ):    
    
    def __init__(self,
        no_cuda=True,
        parallel=False,
        seed=1,
        gpu=0
        ):
        super(NeuralNetEnsamble, self).__init__()
        
        self.pool = PoolNet()
        self.cuda = not no_cuda
        self.parallel = parallel
        self.seed = seed
        self.gpu = gpu
        
        
    def create(self, dataprojects, project):        
        pass 
    
    def __len__(self):
        return len( self.pool )
    
    def __call__(self, image):        
        # switch to evaluate mode
        self.pool.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image  
            zs = self.pool(x)
        return zs        
    
    def representation(self, dataloader ):     
        zs = []
        ys = []
        n = len(dataloader)
        self.pool.eval()
        with torch.no_grad():
            for i, sample in enumerate( tqdm( dataloader ) ):
                x, y = sample['image'], sample['label']
                y = y.argmax(1).float()                        
                if self.cuda:
                    x = x.cuda()                
                z = self.pool(x)                
                zs.append( z.data )
                ys.append( y )        
        zs = np.concatenate( zs, axis=0 )
        ys = np.concatenate( ys, axis=0 )                
        return zs, ys



class ExpertNet( nn.Module ):    
    
    def __init__(self, ens, num_classes=10, dim=3, num_channels=3 ):
        super(ExpertNet, self).__init__()
        
        self.dim=dim
        self.ens = ens
        self.expert = None
        self.cls = None
        self.out_expert = len(ens)
        self.size_input = 0
                
        self.cnn = nn.Sequential(
            #nn.Conv2d(num_channels, 6, 3, stride=1, padding=1),
            #nn.MaxPool2d(2, 2),
            #nn.Conv2d(6, 16, 3, stride=1, padding=0),
            #nn.MaxPool2d(2, 2)
            
            nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
        )       
        self.exp = nn.Linear( 256 , self.out_expert * 32 )     #3600       
        self.cls = nn.Linear(self.dim, num_classes )
        
        
        
    def forward(self, x):  
        
        x_ens = self.ens(x)                         #[n, dim, m]      
        x_exp = self.cnn(x)                
        x_exp = x_exp.view(x_exp.shape[0], -1 )     #[n, m]         
        x_exp = self.exp( x_exp )        
        #x_exp = F.sigmoid( x_exp )
        x_exp = F.relu( x_exp )  
        
        x_exp = x_exp.view(x_exp.shape[0], 32, -1 ) #[n, dim, m]           
        #x_exp = x_exp.unsqueeze(dim=1)             #[n, 1, m]                
        x = (x_ens * x_exp).sum(dim=2)              #[n, dim]
        x = self.cls(x)
                
        return x
        
        

      
