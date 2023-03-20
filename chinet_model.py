'''
Description: 
Author: haotian
Date: 2023-03-17 10:54:58
LastEditTime: 2023-03-18 09:49:25
LastEditors:  
'''

import torch
import torch.nn as nn 


class RDB(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.cx2conv = nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.dal_conv_1 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), dilation=2, padding='same')

        self.dal_conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), dilation=2, padding='same')

    def forward(self, imgs):
        imgs = self.cx2conv(imgs)
        h = imgs

        imgs = self.conv(imgs)
        imgs = self.bn(imgs)
        imgs = self.relu(imgs)

        imgs = self.dal_conv_1(imgs)
        imgs = self.bn(imgs)
        imgs = self.relu(imgs)

        imgs = self.dal_conv_2(imgs)

        return h + imgs
    

class RB(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(2, 2), stride=(2,2))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.conv_1 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1, 1), padding='same')
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1, 1), padding='same')
        self.conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1, 1), padding='same')

        self.conv1x1 = nn.Conv2d(out_ch*2, out_ch, kernel_size=(1,1), stride=(1,1))

    def forward(self, imgs, features):
        
        imgs = self.up(imgs)
        imgs = torch.cat([imgs, features], dim=1)

        

        imgs = self.conv1x1(imgs)

        h = imgs

        imgs = self.bn(imgs)
        imgs = self.relu(imgs)
        imgs = self.conv_1(imgs)

        imgs = self.bn(imgs)
        imgs = self.relu(imgs)
        imgs = self.conv_2(imgs)

        imgs = self.bn(imgs)
        imgs = self.relu(imgs)
        imgs = self.conv_3(imgs)

        return h + imgs


class DDC(nn.Module):
    def __init__(self,out_ch) -> None:
        super().__init__()

        self.r1_conv_1 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

        self.r1_conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

        self.r1_conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

        self.r2_conv = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same', dilation=2)

        self.r3_conv_1 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same', dilation=3)

        self.r3_conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same', dilation=3)

        self.r3_conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same', dilation=3)

        self.conv = nn.Conv2d(out_ch*4, out_ch, kernel_size=(1, 1), stride=(1, 1), padding='same')

    def forward(self, imgs):

        t1 = self.r1_conv_1(imgs)
        t2 = self.r3_conv_1(imgs)

        t3 = self.r3_conv_2(self.r1_conv_2(imgs))
        t4 = self.r3_conv_3(self.r2_conv(self.r1_conv_3(imgs)))

        t = torch.cat([t1, t2, t3, t4], dim=1)

        # print(t.shape)

        return self.conv(t)
    

class SRP(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        
        self.relu = nn.ReLU()

        self.conv_1 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

        self.conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

        self.pool_1 = nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=0)

        self.pool_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=(5,5), stride=(1,1), padding=2)

    def forward(self, imgs):

        imgs = self.relu(imgs)

        t_1 = self.conv_1(self.pool_1(imgs))

        imgs = imgs + t_1

        t_2 = self.conv_2(self.pool_2(imgs))

        imgs = imgs + t_2 

        t_3 = self.conv_3(self.pool_3(imgs))

        imgs = imgs + t_3

        return imgs
    

class Stem(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), stride=(1,1), padding='same')

    def forward(self, imgs):

        imgs = self.conv_1(imgs)
        h = imgs
        imgs = self.bn(imgs)
        imgs = self.relu(imgs)
        imgs = self.conv_2(imgs)

        return imgs + h


class Encoder(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()

        self.stem = Stem(in_ch, 32)
        self.maxpoool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.rdb_1 = RDB(32, 64)
        self.rdb_2 = RDB(64, 128)
        self.rdb_3 = RDB(128, 256)
        self.rdb_4 = RDB(256, 512)

        self.ddc = DDC(512)
        self.srp = SRP(512)

        # self.up = nn.Upsample(scale_factor=2)

    def forward(self, imgs):

        imgs = self.stem(imgs)
        imgs = self.maxpoool(imgs)

        imgs = self.rdb_1(imgs)
        t_1 = imgs
        imgs = self.maxpoool(imgs)

        imgs = self.rdb_2(imgs)
        t_2 = imgs
        imgs = self.maxpoool(imgs)

        imgs = self.rdb_3(imgs)
        t_3 = imgs
        imgs = self.maxpoool(imgs)

        imgs = self.rdb_4(imgs)


        return imgs, t_1, t_2, t_3
    

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.rb_1 = RB(in_ch, in_ch//2)

        self.rb_2 = RB(in_ch//2, in_ch//4)

        self.rb_3 = RB(in_ch//4, in_ch//8)

        self.up = nn.ConvTranspose2d(in_ch//8, in_ch//16, kernel_size=(2,2), stride=(2,2))

        self.conv1x1 = nn.Conv2d(in_ch//16, out_ch, kernel_size=(1,1), stride=(1,1))

        self.sigmoid = nn.Sigmoid()


    def forward(self, imgs, features):

        imgs = self.rb_1(imgs, features[2])
        imgs = self.rb_2(imgs, features[1])
        imgs = self.rb_3(imgs, features[0])

        imgs = self.up(imgs)
        imgs = self.conv1x1(imgs)

        imgs = self.sigmoid(imgs)
        
        return imgs
    

class CHINet(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        # self.stem = Stem(in_ch, 32)

        self.encoder = Encoder(in_ch)
        self.decoder = Decoder(512, out_ch)

        self.ddc = DDC(512)
        self.srp = SRP(512)

    def forward(self, imgs):

        # encoder 里有 stem了
        # imgs = self.stem(imgs)
        
        imgs, *features = self.encoder(imgs)
        
        imgs = self.ddc(imgs)
        imgs = self.srp(imgs)

        imgs = self.decoder(imgs, features)

        return imgs
    


    
if __name__ == '__main__':

    # x = torch.randn(1,2,128,128)
    # t = RB(2, 1)
    # print(t(x, torch.randn(1,1,256,256)).shape)

    model = CHINet(1, 1)
    x = torch.randn(1,1, 512, 512)
    print(model(x).shape)



