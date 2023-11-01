import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self, frames=128):
        super(AutoEncoder, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 32, [3,3,3],stride=1, padding=1,padding_mode="replicate"),
            #nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
        )


        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3,3,3],stride=1, padding=1,padding_mode="replicate"),
            #nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3,3,3],stride=1, padding=1,padding_mode="replicate"),
            #nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.ConvUpBlock3 = nn.Sequential(
            nn.Conv3d(64, 32, [3,3,3],stride=1, padding=1,padding_mode="replicate"),
            #nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
        )

        self.ConvUpBlock1 = nn.Sequential(
            nn.Conv3d(32, 3, [3,3,3],stride=1, padding=1,padding_mode="replicate"),
        )

        self.Maxpool = nn.AvgPool3d((1, 2, 2), stride=(1,2, 2))
        self.upnnSpa = nn.Upsample(scale_factor=(1,2,2),mode="nearest")
        self.tanh = nn.Tanh()



    def forward(self, x):
        x_visual = x
        #[batch,channel,width,height] = x.shape # x [3, T, 128,128]

        x = self.ConvBlock1(x)		     # x [16, T, 128,128]
        #x = self.ConvBlock2(x)		     # x [16, T, 128,128]
        x1 = x.clone()                               #skip1
        x = self.Maxpool(x)       # x [16, T, 64,64]


        x = self.ConvBlock3(x)		    # x [32, T, 64,64]
        #x = self.ConvBlock4(x)	    	# x [64, T, 64,64]
        x2 = x.clone()                               #skip1
        x = self.Maxpool(x)                      #skip2

        x = self.ConvBlock5(x)	    	# x [64, T, 64,64]

        x = self.upnnSpa(x)+x2
        x = self.ConvUpBlock3(x)

        x = self.upnnSpa(x)+x1
        x = self.ConvUpBlock1(x)
        x = self.tanh(x)

        return x
