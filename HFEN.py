import torch
import torch.nn as nn

class HFEN(nn.Module):   
    def __init__(self, in_channels, out_channels, num_filters=64):
        super(HFEN, self).__init__()

        self.encoder1 = self.build_encoder(in_channels, num_filters)
        self.encoder2 = self.build_encoder(num_filters, num_filters * 2)
        self.encoder3 = self.build_encoder(num_filters * 2, num_filters * 4)
        

        self.decoder1 = self.build_decoder(num_filters * 4, num_filters * 2)
        self.decoder2 = self.build_decoder(num_filters * 2, num_filters)
        

        self.final_conv = self.build_decoder(num_filters, out_channels)

        self.key1 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=1,padding=0)
        self.key2 = nn.Conv2d(num_filters*4, num_filters*2, kernel_size=1,padding=0)
        self.prelu = nn.PReLU()
        
    def build_encoder(self, in_channels, out_channels):
        encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return encoder

    def build_decoder(self, in_channels, out_channels):
        decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 
            nn.PReLU()
        )
        return decoder
    

    def forward(self, x):

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        
        x = self.decoder1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.key2(x)
        x = self.prelu(x)
        x = self.decoder2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.key1(x)
        x = self.prelu(x)
        x = self.final_conv(x)
        
        return x