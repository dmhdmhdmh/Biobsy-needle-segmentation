import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

__all__ = ['UNet', 'NestedUNet', 'EURC', 'ResNet34Unet', 'AUnet','RAUnet','RAUnet1']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out



class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)



    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_activate=False):
        super(residual_block,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels//4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels//4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels//4, out_channels//4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels//4)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.batch_activate = batch_activate


    def forward(self, x):
        x = self.bn1(x)
        #x = self.relu1(x)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        #if self.batch_activate:
        #    out = self.bn3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out = out + x
        out = F.relu(out)
        return out


class EURC(nn.Module):
    def __init__(self, n_classes, n_channels=3 ,  deep_supervision=False, **kwargs):
        super(EURC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.conv_1 = nn.Conv2d(3, nb_filter[0], 1, padding=0)
        self.resblock1_1 = residual_block(nb_filter[0],nb_filter[0],batch_activate=False)
        self.resblock1_2 = residual_block(nb_filter[0], nb_filter[0], batch_activate=True)
        self.conv_2 = nn.Conv2d(nb_filter[0], nb_filter[1], 1, padding=0)
        self.resblock2_1 = residual_block(nb_filter[1], nb_filter[1], batch_activate=False)
        self.resblock2_2 = residual_block(nb_filter[1], nb_filter[1], batch_activate=True)
        self.conv_3 = nn.Conv2d(nb_filter[1], nb_filter[2], 1, padding=0)
        self.resblock3_1 = residual_block(nb_filter[2], nb_filter[2], batch_activate=False)
        self.resblock3_2 = residual_block(nb_filter[2], nb_filter[2], batch_activate=True)
        self.conv_4 = nn.Conv2d(nb_filter[2], nb_filter[3], 1, padding=0)
        self.resblock4_1 = residual_block(nb_filter[3], nb_filter[3], batch_activate=False)
        self.resblock4_2 = residual_block(nb_filter[3], nb_filter[3], batch_activate=True)
        self.conv_5 = nn.Conv2d(nb_filter[3], nb_filter[4], 1, padding=0)
        self.resblock5_1 = residual_block(nb_filter[4], nb_filter[4], batch_activate=False)
        self.resblock5_2 = residual_block(nb_filter[4], nb_filter[4], batch_activate=True)

        self.up1 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2, padding=0)
        self.up2 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2, padding=0)
        self.up3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, padding=0)
        self.up4 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, padding=0)
        '''
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        '''

        self.conv_6 = nn.Conv2d(nb_filter[4], nb_filter[3], 1, padding=0)
        self.conv_7 = nn.Conv2d(nb_filter[3], nb_filter[2], 1, padding=0)
        self.conv_8 = nn.Conv2d(nb_filter[2], nb_filter[1], 1, padding=0)
        self.conv_9 = nn.Conv2d(nb_filter[1], nb_filter[0], 1, padding=0)
        self.outconv = nn.Conv2d(nb_filter[0], n_classes, 1, padding=0)


    def forward(self, x):
        x0_1 = self.conv_1(x)
        x0_2 = self.resblock1_1(x0_1)
        x0_3 = self.resblock1_2(x0_2)
        x0_4 = self.pool(x0_3)
        x0_4 = self.dropout(x0_4)
        x0_5 = self.conv_2(x0_4)
        x0_6 = self.resblock2_1(x0_5)
        x0_7 = self.resblock2_2(x0_6)
        x0_8 = self.pool(x0_7)
        x0_8 = self.dropout(x0_8)
        x0_9 = self.conv_3(x0_8)
        x0_10 = self.resblock3_1(x0_9)
        x0_11 = self.resblock3_2(x0_10)
        x0_12 = self.pool(x0_11)
        x0_12 = self.dropout(x0_12)
        x0_13 = self.conv_4(x0_12)
        x0_14 = self.resblock4_1(x0_13)
        x0_15 = self.resblock4_2(x0_14)
        x0_16 = self.pool(x0_15)
        x0_16 = self.dropout(x0_16)
        x0_17 = self.conv_5(x0_16)
        x0_18 = self.resblock5_1(x0_17)
        x0_19 = self.resblock5_2(x0_18)
        up1 = torch.cat([x0_15, self.up1(x0_19)], 1)
        up1 = self.dropout(up1)
        x1_1 = self.conv_6(up1)
        x1_2 = self.resblock4_1(x1_1)
        x1_3 = self.resblock4_2(x1_2)
        up2 = torch.cat([x0_11, self.up2(x1_3)], 1)
        up2 = self.dropout(up2)
        x1_4 = self.conv_7(up2)
        x1_5 = self.resblock3_1(x1_4)
        x1_6 = self.resblock3_2(x1_5)
        up3 = torch.cat([x0_7, self.up3(x1_6)], 1)
        up3 = self.dropout(up3)
        x1_7 = self.conv_8(up3)
        x1_8 = self.resblock2_1(x1_7)
        x1_9 = self.resblock2_2(x1_8)
        up4 = torch.cat([x0_3, self.up4(x1_9)], 1)
        up4 = self.dropout(up4)
        x1_10 = self.conv_9(up4)
        x1_11 = self.resblock1_1(x1_10)
        x1_12 = self.resblock1_2(x1_11)
        out = self.outconv(x1_12)
        return out

class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=3,
                 isdeconv=False
                 ):
        super().__init__()


        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               kernel_size,
                               padding=conv_padding, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if isdeconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1, bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               kernel_size,
                               padding=conv_padding, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet34Unet(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 deep_supervision=False,
                 **kwargs
                 ):
        super().__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.base_size = 512
        self.crop_size = 512
        self.isdeconv = False
        self.decoderkernel_size = 3
        #self.isdeconv = bool(False),
        #self.decoderkernel_size = int(3),
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=self.decoderkernel_size,
                                   isdeconv=self.isdeconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)

        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        f = self.finalconv(d1)
        return f


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class AUnet(nn.Module):
    def __init__(self, output_ch=1,img_ch=3,deep_supervision=False,**kwargs):
        super(AUnet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

class RAUnet(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 deep_supervision=False,
                 **kwargs
                 ):
        super().__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.base_size = 512
        self.crop_size = 512
        self.isdeconv = False
        self.decoderkernel_size = 3
        #self.isdeconv = bool(False),
        #self.decoderkernel_size = int(3),
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[3], F_int=filters[2])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[2], F_int=filters[1])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[1], F_int=filters[0])
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0])
        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=self.decoderkernel_size,
                                   isdeconv=self.isdeconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)

        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x0 = self.firstconv(x)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x_ = self.firstmaxpool(x0)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)
        e3 = self.Att4(g=center, x=e3)
        d4 = self.decoder4(torch.cat([center, e3], 1))
        e2 = self.Att3(g=d4, x=e2)
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        e1 = self.Att2(g=d3, x=e1)
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        x0 = self.Att1(g=d2, x=x0)
        d1 = self.decoder1(torch.cat([d2, x0], 1))

        f = self.finalconv(d1)
        return f

class RAUnet1(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 deep_supervision=False,
                 **kwargs
                 ):
        super().__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.base_size = 512
        self.crop_size = 512
        self.isdeconv = False
        self.decoderkernel_size = 1
        #self.isdeconv = bool(False),
        #self.decoderkernel_size = int(3),
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[3], F_int=filters[2])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[2], F_int=filters[1])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[1], F_int=filters[0])
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0])
        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=self.decoderkernel_size,
                                   isdeconv=self.isdeconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=self.decoderkernel_size,
                                     isdeconv=self.isdeconv)

        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x0 = self.firstconv(x)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x_ = self.firstmaxpool(x0)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)
        e3 = self.Att4(g=center, x=e3)
        d4 = self.decoder4(torch.cat([center, e3], 1))
        e2 = self.Att3(g=d4, x=e2)
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        e1 = self.Att2(g=d3, x=e1)
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        x0 = self.Att1(g=d2, x=x0)
        d1 = self.decoder1(torch.cat([d2, x0], 1))

        f = self.finalconv(d1)
        return f