import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        ## encoder
        self.enc1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 이미지 H,W size를 2배 감소

        self.enc2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.enc5_2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
      
    
        ## decoder
        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) # 이미지 H,W size를 2배 늘린다.
        self.dec4_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec4_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec3_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.dec1_3 = nn.Conv2d(64, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x): # 3*256*256(C*H*W)
        ## encoder
        x = self.enc1_1(x) # 64*256*256
        x = self.relu(x)
        e1 = self.enc1_2(x) # 64*256*256
        e1 = self.relu(e1)
        # decoder에서 e1이 skip connection으로 사용됨
        x = self.pool1(e1) # 64*128*128

        x = self.enc2_1(x) # 128*128*128
        x = self.relu(x)
        e2 = self.enc2_2(x) # 128*128*128
        e2 = self.relu(e2)
        x = self.pool2(e2) # 128*64*64

        x = self.enc3_1(x) # 256*64*64
        x = self.relu(x)
        e3 = self.enc3_2(x) # 256*64*64
        e3 = self.relu(e3)
        x = self.pool3(e3) # 256*32*32

        x = self.enc4_1(x) # 512*32*32
        x = self.relu(x)
        e4 = self.enc4_2(x) # 512*32*32
        e4 = self.relu(e4)
        x = self.pool4(e4) # 512*16*16

        x = self.enc5_1(x) # 1024*16*16
        x = self.relu(x)
        x = self.enc5_2(x) # 512*16*16
        x = self.relu(x)

        ##decoder
        x = self.upsample4(x) # 512*32*32
        # e4와 x를 residual connection 진행 ,, dim=1 >> channel 방향
        x = torch.cat([x, e4], dim=1) # 1024*32*32
        x = self.dec4_1(x) # 512*32*32
        x = self.relu(x)
        x = self.dec4_2(x) # 256*32*32
        x = self.relu(x)

        x = self.upsample3(x) # 256*64*64
        x = torch.cat([x, e3], dim=1) # 512*64*64
        x = self.dec3_1(x) # 256*64*64
        x = self.relu(x)
        x = self.dec3_2(x) # 128*64*64
        x = self.relu(x)

        x = self.upsample2(x) # 128*128*128
        x = torch.cat([x, e2], dim=1) # 256*128*128
        x = self.dec2_1(x) # 128*128*128
        x = self.relu(x)
        x = self.dec2_2(x) # 64*128*128
        x = self.relu(x)

        x = self.upsample1(x) # 64*256*256
        x = torch.cat([x, e1], dim=1) # 128*256*256
        x = self.dec1_1(x) # 64*256*256
        x = self.relu(x)
        x = self.dec1_2(x) # 64*256*256
        x = self.relu(x)
        
        x = self.dec1_3(x) # 1*256*256
        # tensor x에서 값이 1인 차원을 제거 >> 흑백으로 만듬
        x = torch.squeeze(x) # 256*256

        return x