
# STD MODULES 
import os
import math
import shutil
import time
import numpy as np
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# TORCH MODULE
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn

# PYTVISION MODULE
from pytvision.neuralnet import NeuralNetAbstract
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import utils as pytutils
from pytvision import graphic as gph
from pytvision import netlearningrate

#LOCAL MODULE
from . import models as embnn
from . import netlosses as losses
from . import tripletnet as tnn



class NeuralNet(NeuralNetAbstract):
    r"""NeuralNet Model in embedded space 
    Args:
        patchproject (str): path project
        nameproject (str):  name project
        no_cuda (bool): system cuda (default is True)
        parallel (bool):
        seed (int):
        print_freq (int):
        gpu (int):
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        super(NeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )

    def create(self, 
        arch, 
        num_output_channels, 
        num_input_channels,  
        loss, 
        lr,  
        optimizer, 
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,          
        pretrained=False,
        classes=10,
        size_input=128,
        ):
        """
        Create
        Args:
            arch (string): architecture
            num_output_channels, 
            num_input_channels,  
            loss (string):
            lr (float): learning rate
            momentum,
            optimizer (string) : 
            lrsch (string): scheduler learning rate
            pretrained (bool)
        """

        self.size_input = size_input
        self.num_classes = classes
        cfg_opt={ 'momentum':momentum, 'weight_decay':weight_decay } 
        cfg_scheduler={ 'step_size':100, 'gamma':0.1  }
        

        super(NeuralNet, self).create( 
            arch, 
            num_output_channels, 
            num_input_channels, 
            loss, 
            lr, 
            optimizer, 
            lrsch, 
            pretrained,
            cfg_opt=cfg_opt, 
            cfg_scheduler=cfg_scheduler,
            )    

    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained):
        """
        Create model
            arch (string): select architecture
            num_input_channels (int)
            num_output_channels (int)
            pretrained (bool)
        """    

        self.fcn = None
        self.net = None
        
        kw = {'dim': num_output_channels, 'num_channels': num_input_channels, 'pretrained': pretrained}
        self.fcn = embnn.__dict__[arch](**kw)
        self.net = tnn.Tripletnet( self.fcn )
        
        self.s_arch = arch 
        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels
        self.pretrained = pretrained

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))
 


    def save(self, epoch, prec, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save model
        """
        print('>> save model epoch {} ({}) in {}'.format(epoch, prec, filename))
        net = self.net.module if self.parallel else self.net
        pytutils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.s_arch,
                'imsize': self.size_input,
                'num_output_channels': self.num_output_channels,
                'num_input_channels': self.num_input_channels,
                'num_classes': self.num_classes,
                'state_dict': net.state_dict(),
                'prec': prec,
                'optimizer' : self.optimizer.state_dict(),
            }, 
            is_best,
            self.pathmodels,
            filename
            )

    def load(self, pathnamemodel):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )                
                self.num_classes = checkpoint['num_classes']
                self._create_model(checkpoint['arch'], checkpoint['num_output_channels'], checkpoint['num_input_channels'], False )                
                self.size_input = checkpoint['imsize'] 
                self.net.load_state_dict( checkpoint['state_dict'] )              
                print("=> loaded checkpoint for {} arch!".format(checkpoint['arch']))
                bload = True
            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))        
        return bload            



class TripletNeuralNet(NeuralNet):
    """
    Triplet Neural Net Class
    """


    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel = False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        """
        Initialization
            -patchproject (str)
            -nameproject (str)
            -no_cuda (bool) (default is True)
            -seed (int)
            -print_freq (int)
            -gpu (int)
        """

        super(TripletNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )
        
        # Set the graphic visualization
        self.fcn = None


       
    def create(
        self,
        arch,
        num_output_channels, 
        num_input_channels,  
        loss, 
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,          
        pretrained=False,
        classes=10,
        size_input=128,
        margin=1,
        ):
        """
        Create
        Args:
            arch (string): architecture
            num_output_channels, 
            num_input_channels,  
            loss (string):
            lr (float): learning rate
            momentum,
            optimizer (string) : 
            lrsch (string): scheduler learning rate
            pretrained (bool)
        """
       
        self.margin = margin
        super(TripletNeuralNet, self).create( 
            arch, 
            num_output_channels, 
            num_input_channels, 
            loss, 
            lr, 
            optimizer, 
            lrsch, 
            momentum,
            weight_decay,
            pretrained,
            classes,
            size_input,
            )
        
        
        self.accuracy = losses.Accuracy( )

        # Set the graphic visualization
        self.logger_train = Logger( 'Trn', ['loss'], ['acc'], self.plotter  )
        self.logger_val   = Logger( 'Val', ['loss'], ['acc'], self.plotter )



    def training(self, data_loader, epoch=0):

        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, sample in enumerate(data_loader):
           
            # measure data loading time
            data_time.update(time.time() - end)

            anch = sample['b']; data_anch = pytutils.to_var(anch['image'], self.cuda)
            pos  = sample['a']; data_pos  = pytutils.to_var( pos['image'], self.cuda)
            neg  = sample['c']; data_neg  = pytutils.to_var( neg['image'], self.cuda)  
            batch_size = data_pos.size(0)     
            
            # compute output
            embedded_a, embedded_p, embedded_n = self.net(data_anch, data_pos, data_neg)
            target = torch.FloatTensor(embedded_a.size()[1]).fill_(1)
            target = pytutils.to_var( target, self.cuda)

            # measure accuracy and record loss
            loss_triplet = self.criterion(embedded_a, embedded_p, embedded_n, target)
            loss_embedd = embedded_a.norm(2) + embedded_p.norm(2) + embedded_n.norm(2)
            loss = loss_triplet + 0.001 * loss_embedd
            acc = self.accuracy(embedded_a, embedded_p, embedded_n)
              
            # optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update
            self.logger_train.update(
                {'loss': loss.data[0] },
                {'acc': acc },
                batch_size,
                )       

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )
    
  
    def evaluate(self, data_loader, epoch=0):
        
        self.logger_val.reset()
        batch_time = AverageMeter()
        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate(data_loader):
            
                anch = sample['b']; data_anch = pytutils.to_var(anch['image'], self.cuda, False, True)
                pos  = sample['a']; data_pos  = pytutils.to_var( pos['image'], self.cuda, False, True)
                neg  = sample['c']; data_neg  = pytutils.to_var( neg['image'], self.cuda, False, True)       
                batch_size = data_pos.size(0)       
                
                # compute output
                embedded_a, embedded_p, embedded_n = self.net(data_anch, data_pos, data_neg)                                       

                # 1 means, dista should be larger than distb
                target = torch.FloatTensor(embedded_a.size()[1]).fill_(1)
                target = pytutils.to_var( target, self.cuda)

                # measure accuracy and record loss
                loss_triplet = self.criterion(embedded_a, embedded_p, embedded_n, target)
                loss_embedd = embedded_a.norm(2) + embedded_p.norm(2) + embedded_n.norm(2)
                loss = loss_triplet + 0.001 * loss_embedd
                acc = self.accuracy(embedded_a, embedded_p, embedded_n)

                # update
                self.logger_val.update(
                {'loss': loss.data[0] },
                {'acc': acc },
                batch_size,
                )

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['acc'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        return acc


    def __call__(self, image):

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image  
            emb = self.fcn(x)
        return emb

    
    
    def representation(self, dataloader ):           
        zs = []
        ys = []
        n = len(dataloader)
        self.net.eval()
        with torch.no_grad():
            for i, sample in enumerate( tqdm( dataloader ) ):
                x, y = sample['image'], sample['label']
                y = y.argmax(1).float()                        
                if self.cuda:
                    x = x.cuda()
                z = self.fcn(x)                
                zs.append( z.data.cpu() )
                ys.append( y )   
        
        zs = np.concatenate( zs, axis=0)
        ys = np.concatenate( ys, axis=0)                
        return zs, ys
    

    def inference(self):
        pass       
    
    def _to_end_epoch(self, epoch, epochs, train_loader, val_loader):
        train_loader.dataset.reset()
        val_loader.dataset.reset()



    def _create_loss(self, loss):
        """
        Create loss
            -loss (str): type loss
            -margin (float): margin
        """       

        # loss to use
        if loss == 'hinge':
            self.criterion = losses.EmbHingeLoss( margin=self.margin )
        elif loss == 'square':
            self.criterion = losses.EmbSquareHingeLoss( margin=self.margin ) 
        elif loss == 'soft':
            self.criterion = losses.EmbSoftHingeLoss( margin=self.margin ) 
        else:
            assert(False)
        
        self.loss = loss


