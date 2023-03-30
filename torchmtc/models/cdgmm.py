#----------------------------------------------------------------------------------------------
# CGMMAENet
# Pedro D. Marrero Fernandez
#----------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = ['CDGMMAENet', 'cdgmm']



class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, bn=False):
        super(ResBlock, self).__init__()
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0)
            ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(channels))
        self.convs = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.convs(x)  
 
class EncoderNet(nn.Module):
    """
    Encoder Convolutional Net
    """
    def __init__(self, d, bn=True, num_channels=3, **kwargs):
        super(EncoderNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn),
            nn.BatchNorm2d(d),
        )
    def forward(self, x):           
        return self.encoder(x)

class DecoderNet(nn.Module):
    """
    Decoder Convolutional Net
    """
    def __init__(self, d, bn=True, num_channels=3, **kwargs):
        super(DecoderNet, self).__init__()

        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):         
        return F.tanh(self.decoder(x))


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class EncoderResNet(nn.Module):
    """
    Encoder Convolutional Net
    """
    def __init__(self, d, bn=True, num_channels=3, pretrained=True, **kwargs):
        super(EncoderResNet, self).__init__()

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        bottom_channel_nr = 512
        
        #self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            #self.pool
        )

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.conv6 = conv3x3(bottom_channel_nr, d)
        
        
    def forward(self, x):          
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)        
        #pool = self.pool(conv5)
        enc = conv6            
        return enc 

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels ):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels
        
        self.block = nn.Sequential(
            ResBlock(in_channels, in_channels, True),
            ConvRelu(in_channels, middle_channels),
            nn.BatchNorm2d(middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
    
class DecoderResNet(nn.Module):
    """
    Decoder Convolutional Net
    """
    def __init__(self, d, bn=True, num_channels=3, **kwargs):
        super(DecoderResNet, self).__init__()

        self.decoder = nn.Sequential(
            #DecoderBlockV2(d, d*8, d*8 ),
            DecoderBlockV2(d, d*4, d*4 ),
            DecoderBlockV2(d*4, d*2, d*2 ),
            DecoderBlockV2(d*2, d, d ),
            #DecoderBlockV2(d, num_channels, num_channels ),
            ConvRelu(d, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=4, stride=2, padding=1),            
            
        )

    def forward(self, x):         
        return F.tanh( self.decoder(x) )

class CDGMMAENet(nn.Module):
    def __init__(self, dim=64, num_channels=3, bn=True, pretrained=True, **kwargs):
        super(CDGMMAENet, self).__init__()

        self.encoder = EncoderNet(dim, bn, num_channels, pretrained=True ) #pretrained
        self.decoder = DecoderNet(dim, bn, num_channels )
        
        #for l in self.modules():
        #    if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
        #        l.weight.detach().normal_(0, 0.02)
        #        torch.fmod(l.weight, 0.04)
        #        nn.init.constant_(l.bias, 0)
        #self.encoder.encoder[-1].weight.detach().fill_(1 / 40)

    def forward(self, x):      
        z = self.encoder(x)   
        return self.decoder(z), z.view( x.shape[0], -1 ) #<- xp, z



def cdgmm(pretrained=False, **kwargs):
    r"""Simple model architecture for embedded models
    """
    model = CDGMMAENet(pretrained=pretrained, **kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['cdgmm']))
    return model
    



def test():
    
    net = CDGMMAENet( dim=64, bn=True, num_channels=3 )    
    x_preds, z = net( torch.randn( 10, 3, 128, 128 ) )
    
    print(x_preds.size())
    print(z.size())

    x_grads = torch.randn(x_preds.shape)
    z_grads = torch.randn(z.shape)
    x_preds.backward(x_grads, retain_graph=True)
    z.backward(z_grads)

    
def test_res_enc():
    
    net_enc = EncoderResNet(dim=64 , bn=True, num_channels=3, pretrained=True)
    net_dec = DecoderResNet(dim=64 , bn=True, num_channels=3)
    
    x = torch.randn( 10, 3, 64, 64 )
    print(x.shape)    
    z = net_enc( x )
    print(z.shape)
    x = net_dec( z )
    print(x.shape)
    

# test()
# test_res_enc()