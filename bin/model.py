import torch
import torch.nn as nn
from torch.nn import init

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, seq_length=150):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.doubleConv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=7, padding=3), # The sequence length will remain the same
            nn.LayerNorm(normalized_shape=[mid_channels, seq_length]),
            #nn.BatchNorm1d(mid_channels),
            #nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=7, padding=3), # The sequence length will remain the same
            #nn.BatchNorm1d(out_channels),
            nn.GELU()
            #nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.doubleConv(x)


class Down(nn.Module):
    """Downsampling using maxpooling, followed by 2 layers of convolution"""
    def __init__(self, in_channels, out_channels, mid_channels=None, special=False, seq_length=150):
        super().__init__()
        if not special:
            self.maxpoolConv = nn.Sequential(
                nn.MaxPool1d(2),
                DoubleConv(in_channels, out_channels, mid_channels, seq_length)
            )
        else:
            self.maxpoolConv = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1), 
                DoubleConv(in_channels, out_channels, mid_channels, seq_length)
            )
        
    def forward(self, x):
        return self.maxpoolConv(x)


class Up(nn.Module):
    """Upsampling (with skip connection) then 2 conv layers"""
    def __init__(self, in_channels, out_channels, mid_channels=None, special=False, seq_length=150):
        super().__init__()
        
        if not special:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2) 
            self.conv = DoubleConv(in_channels, out_channels, seq_length=seq_length)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1) #[256, 38] -> [128, 75]
            self.conv = DoubleConv(in_channels, out_channels, seq_length=seq_length) #[256, 38] -> [128, 75]
            
    def forward(self, x1, x2):
        x1 = self.up(x1) # [128, 75]
        x = torch.cat([x2, x1], dim=1) # [256, 75]
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, 64)
        self.dropout1 = nn.Dropout()
        self.relu1 = nn.GELU()
        self.linear2 = nn.Linear(64, out_dim)
        self.dropout2 = nn.Dropout()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class Unet1D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet1D, self).__init__()
                
        self.inConv = DoubleConv(n_channels, 32) # (150+2*1-3)/1+1 = 150 (4 -> 32)
        self.selayer1 = SELayer(32)
        self.down1 = Down(32, 64, seq_length=75) # (Maxpooling -> 75) (32 -> 64)
        self.selayer2 = SELayer(64)
        self.down2 = Down(64, 128, special=True, seq_length=38) # (Maxpooling -> 38) (64 -> 128)
        self.selayer3 = SELayer(128)
        self.down3 = Down(128, 256, seq_length=19) # (Maxpooling -> 19) (128 -> 256)
        
        self.up1 = Up(256, 128, seq_length=38) # (ConvTranspose -> 38) (256 -> 128)
        self.up2 = Up(128, 64, special=True, seq_length=75) # (ConvTranspose -> 75) (128 -> 64)
        self.up3 = Up(64, 32) # (ConvTranspose -> 150) (64 -> 32)
        self.outConv = OutConv(32, n_classes)
        self.classConv = MLP(150, 1)
        
    def forward(self, x):
        x1 = self.inConv(x) # [x, 32, 150]
        x1 = self.selayer1(x1)
        x2 = self.down1(x1) # [x, 64, 75]
        x2 = self.selayer2(x2)
        x3 = self.down2(x2) # [x, 128, 38]
        x3 = self.selayer3(x3)
        x4 = self.down3(x3) # [x, 256, 19]
        
        x = self.up1(x4, x3) # [x, 128, 38]
        x = self.up2(x, x2) # [x, 64, 75]
        x = self.up3(x, x1) # [x, 32, 150]
        out = self.outConv(x) # [x, 1, 150]
        #out = torch.sigmoid(out)
        #out_conf = self.classConv(out.view(out.size()[0], -1))
        out_conf = self.classConv(out.squeeze(1))
        return (torch.sigmoid(out), torch.sigmoid(out_conf))
