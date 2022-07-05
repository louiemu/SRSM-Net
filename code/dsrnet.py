import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from base_networks import *
from modules import *
from tuning_blocks import *
from torchvision.transforms import *


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()

        "Set conv-way depend on scale_factor"
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        ####
        elif scale_factor == 16:
            kernel = 20
            stride = 16
            padding = 2

        "SRS"
        self.rgbEncoder = Struct_Encoder(n_downsample=2, n_res=4, 
                                                input_dim=3, dim=64, 
                                                norm='in', activ='lrelu', 
                                                pad_type='reflect')
        self.rgbEncoder.requires_grad = False
        self.rgbDecoder = Struct_Decoder()
        self.output_conv_color = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        self.depth0 = ConvBlock(1, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.depth1 = ResnetBlock(base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.depth2 = ResnetBlock(base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.edge0 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, norm=None)
        self.edge1 = ResnetBlock(base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.edge2 = ResnetBlock(base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.output_conv_e = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        "SRM"
        self.attention_feature1 = ConvBlock(2*base_filter, 1, 3, 1, 1, activation='prelu', norm=None)
        self.attention_feature2 = ConvBlock(2*base_filter, 1, 3, 1, 1, activation='prelu', norm=None)
        self.attention_feature3 = ConvBlock(2*base_filter, 1, 3, 1, 1, activation='prelu', norm=None)
        self.attention_feature4 = ConvBlock(2*base_filter, 1, 3, 1, 1, activation='prelu', norm=None)
        self.attention_feature5 = ConvBlock(2*base_filter, 1, 3, 1, 1, activation='prelu', norm=None)
        self.attention_feature6 = ConvBlock(2*base_filter, 1, 3, 1, 1, activation='prelu', norm=None)
        self.attention_feature7 = ConvBlock(2*base_filter, 1, 3, 1, 1, activation='prelu', norm=None)

        "Main Network"
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.up1 = UpBlock(base_filter, kernel, stride, padding)

        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)

        self.down2 = DownBlock(base_filter, kernel, stride, padding)
        self.up3 = UpBlock(base_filter, kernel, stride, padding)

        self.down3 = DownBlock(base_filter, kernel, stride, padding)
        self.up4 = UpBlock(base_filter, kernel, stride, padding)

        self.down4 = DownBlock(base_filter, kernel, stride, padding)
        self.up5 = UpBlock(base_filter, kernel, stride, padding)
        
        self.down5 = DownBlock(base_filter, kernel, stride, padding)
        self.up6 = UpBlock(base_filter, kernel, stride, padding)
        
        self.down6 = DownBlock(base_filter, kernel, stride, padding)
        self.up7 = UpBlock(base_filter, kernel, stride, padding)

        self.down7 = DownBlock(base_filter, kernel, stride, padding)
        self.up8 = UpBlock(base_filter, kernel, stride, padding)

        self.Bicubic = torch.nn.Upsample(size=None, scale_factor=scale_factor, mode='bicubic', align_corners=None) #???????????????????????????????
        self.output_conv_final = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        # "initialization"
        # for m in self.modules():
        #     classname = m.__class__.__name__
        #     if classname.find('Conv2d') != -1:
        #         torch.nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif classname.find('ConvTranspose2d') != -1:
        #         torch.nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self,rgb, depth):
        "SRS"
        bicubic = self.Bicubic(depth)

        d0 = self.depth0(bicubic)
        d1 = self.depth1(d0)
        depthFeat = self.depth2(d1)
        
        c1 = self.rgbEncoder(rgb)
        colorFeat = self.rgbDecoder(c1)

        colorStructure = torch.sigmoid(self.output_conv_color(colorFeat))

        edge0 = self.edge0(depthFeat*colorFeat)
        edge1 = self.edge1(edge0)
        edgeFeat = self.edge2(edge1)

        edgeMap = torch.sigmoid(self.output_conv_e(edgeFeat))

        "Main Network and SRM"
        x1 = self.feat0(depth)
        x2 = self.feat1(x1)

        h1 = self.up1(x2)
        G1 = self.attention_feature1(torch.cat((edgeFeat,h1),1))
        G1 = F.adaptive_avg_pool2d(torch.sigmoid(G1),1)
        h1 = h1 + G1[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*edgeFeat
        l1 = self.down1(h1)

        h2 = self.up2(l1)
        G2 = self.attention_feature2(torch.cat((edgeFeat,h2),1))
        G2 = F.adaptive_avg_pool2d(torch.sigmoid(G2),1)
        h2 = h2 + G2[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*edgeFeat
        l2 = self.down2(h2)

        h3 = self.up3(l2)
        G3 = self.attention_feature3(torch.cat((edgeFeat,h3),1))
        G3 = F.adaptive_avg_pool2d(torch.sigmoid(G3),1)
        h3 = h3 + G3[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*edgeFeat
        l3 = self.down3(h3)

        h4 = self.up4(l3)
        G4 = self.attention_feature4(torch.cat((edgeFeat,h4),1))
        G4 = F.adaptive_avg_pool2d(torch.sigmoid(G4),1)
        h4 = h4 + G4[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*edgeFeat
        l4 = self.down4(h4)

        h5 = self.up5(l4)
        G5 = self.attention_feature5(torch.cat((edgeFeat,h5),1))
        G5 = F.adaptive_avg_pool2d(torch.sigmoid(G5),1)
        h5 = h5 + G5[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*edgeFeat
        l5 = self.down5(h5)

        h6 = self.up6(l5)
        G6 = self.attention_feature6(torch.cat((edgeFeat,h6),1))
        G6 = F.adaptive_avg_pool2d(torch.sigmoid(G6),1)
        h6 = h6 + G6[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*edgeFeat
        l6 = self.down6(h6)

        h7 = self.up7(l6)
        G7 = self.attention_feature7(torch.cat((edgeFeat,h7),1))
        G7 = F.adaptive_avg_pool2d(torch.sigmoid(G7),1)
        h7 = h7 + G7[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*edgeFeat
        l7 = self.down7(h7)

        h8 = self.up8(l7)

        SR = self.output_conv_final(h8)

        return colorStructure, edgeMap, SR, G1,G2,G3,G4,G5,G6,G7