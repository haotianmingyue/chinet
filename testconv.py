

import torch
import torch.nn as nn

class A(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1,1), padding='same')

    def forward(self, imgs):
        t1 = self.conv(imgs)
        t2 = self.conv(imgs)

        print(t1 == t2)
        return t1 + t2 
    
if __name__ == '__main__':
    x = torch.randn(1,1, 256, 256)
    a = A(1,1)
    print(a(x).shape)