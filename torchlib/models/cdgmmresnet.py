

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision



__all__ = ['CDGMMResNet', 'cdgmmresnet152', 'cdgmmresnet101', 'cdgmmresnet34']



def cdgmmresnet152(pretrained=False, **kwargs):
    """"CDGMMResNet model architecture
    """
    model = CDGMMResNet(encoder_depth=152 ,pretrained=pretrained, **kwargs)

    if pretrained == True:
        #model.load_state_dict(state['model'])
        pass
    return model

def cdgmmresnet101(pretrained=False, **kwargs):
    """"CDGMMResNet model architecture
    """
    model = CDGMMResNet(encoder_depth=101 ,pretrained=pretrained, **kwargs)

    if pretrained == True:
        #model.load_state_dict(state['model'])
        pass
    return model


def cdgmmresnet34(pretrained=False, **kwargs):
    """"CDGMMResNet model architecture
    """
    model = CDGMMResNet(encoder_depth=34 ,pretrained=pretrained, **kwargs)

    if pretrained == True:
        #model.load_state_dict(state['model'])
        pass
    return model


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

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class Conv2D(nn.Module):
    def __init__(self, filtersin, filtersout, kernel_size=(3,3), s=1, pad=0, is_batchnorm=False):
        super(Conv2D, self).__init__()
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(filtersin, filtersout, kernel_size, s, pad), nn.BatchNorm2d(filtersout), nn.ReLU(),)
        else:
            self.conv = nn.Sequential(nn.Conv2d(filtersin, filtersout, kernel_size, s, pad), nn.ReLU(),)
    def forward(self, x):
        x = self.conv(x)
        return x
    
class DilateCenter(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, is_batchnorm=False ):
        super(DilateCenter, self).__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        #self.conv_init = Conv2D( in_size, out_size, kernel_size, s=1, pad=0, is_batchnorm=is_batchnorm )     
        self.conv_d1 = nn.Conv2d(in_size,  out_size, kernel_size, 1, kernel_size//2 + 0, dilation=1 )
        self.conv_d2 = nn.Conv2d(out_size, out_size, kernel_size, 1, kernel_size//2 + 1, dilation=2 )
        self.conv_d3 = nn.Conv2d(out_size, out_size, kernel_size, 1, kernel_size//2 + 2, dilation=3 )
        self.conv_d4 = nn.Conv2d(out_size, out_size, kernel_size, 1, kernel_size//2 + 3, dilation=4 )
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()        
        
        #self.conv_end = Conv2D( out_size, out_size, kernel_size, s=1, pad=0, is_batchnorm=is_batchnorm )        
        #self.W1 = nn.Parameter( torch.rand(embeddings_dim, num_embeddings) )
        
    
    def forward(self, x ): 
        
        skip = torch.zeros( x.shape[0], self.out_size, x.shape[2], x.shape[3] ).cuda()
        x1 = x       
                
        #x1 = self.conv_init(x); skip+=x1
        x2 = self.conv_d1(x1);  skip+= x2
        x3 = self.conv_d2(x2);  skip+= x3 
        x4 = self.conv_d3(x3);  skip+= x4 
        x5 = self.conv_d4(x4);  skip+= x5
        x6 = self.relu (skip)
        #y = self.conv_end(x6);  
        y = x6

        return y
    
    
class _Residual_Block_DB(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_DB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output


class _Residual_Block_SR(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_SR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output


    
class CDGMMResNet(nn.Module):
    """PyTorch CDGMM model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    
    def __init__(self, encoder_depth, dim=32, num_classes=1, num_channels=3, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=True):
        
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        
        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        
        #self.center = DecoderBlockV2( bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.center = DilateCenter( bottom_channel_nr, num_filters * 8 )
                
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        
        #autoendocer
        self.autoencoder = nn.Sequential(
            ConvRelu(num_filters, num_filters),
            nn.Conv2d(num_filters, num_channels, kernel_size=1) 
        )        
        
        self.coder = nn.Linear(bottom_channel_nr//2 , dim)
        
    
    
    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)
        

    def forward(self, x ):
        
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        #pool = self.pool(conv5)
        center = self.center( conv5 )  
                        
        out = F.avg_pool2d(center, center.shape[3])   
        out = out.view(out.size(0), -1)

        z = self.coder(out)
                
        dec5 = self.dec5(torch.cat([center, conv5], 1))        
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        fmap = self.dec1(dec2)   
        
        #autoencoder
        xr = self.autoencoder(fmap)
            
        return xr, z
    



def test():
    
    batch=10
    num_channels=3
    num_classes=10
    dim=20
    
    net = cdgmmresnet34( False, dim=dim, num_channels=num_channels, num_classes=num_classes ).cuda()
    xr, z = net(  torch.randn(batch, num_channels, 64, 64 ).cuda() ) 
        
    print( z.shape )
    print( xr.shape )
    
   


if __name__ == "__main__":
    test()


