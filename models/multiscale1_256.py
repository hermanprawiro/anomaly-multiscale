import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, bias=False, transpose=False, norm='batch', activation='relu'):
        super().__init__()

        if transpose:
            self.conv = nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=bias)
        else:
            self.conv = nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=bias)

        if norm == 'batch':
            self.norm = nn.BatchNorm2d(n_out)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(n_out)
        else:
            self.norm = None
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class EncoderShared(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(4, 1, 1), padding=(1, 0, 0), stride=(2, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(4, 1, 1), padding=(1, 0, 0), stride=(2, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.shared_enc = EncoderShared(in_channels=in_channels)

    def forward(self, x):
        z_scale1 = self.shared_enc(x) # (n, 128, 1, 64, 64)
        z_scale2 = self.shared_enc(nn.functional.interpolate(x, scale_factor=(1, 1/4, 1/4))) # (n, 128, 1, 16, 16)
        z_scale3 = self.shared_enc(nn.functional.interpolate(x, scale_factor=(1, 1/16, 1/16))) # (n, 128, 1, 4, 4)

        z_scale2 = nn.functional.interpolate(z_scale2, scale_factor=(1, 4, 4))
        z_scale3 = nn.functional.interpolate(z_scale3, scale_factor=(1, 16, 16))
        z = torch.cat((z_scale1, z_scale2, z_scale3), 1)

        return z

class Decoder_Background(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super().__init__()

        # Input (n, in_channels, 64, 64)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            nn.Upsample(scale_factor=2), # (n, in_channels, 128, 128)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2), # (n, 64, 256, 256)
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, out_channels, kernel_size=1, padding=0, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.squeeze(2)
        x = self.model(x)
        return x

class Decoder_Foreground(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super().__init__()

        # Input (n, in_channels, 1, 64, 64)
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),

            nn.Upsample(scale_factor=2), # (n, in_channels, 2, 128, 128)
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.mask = nn.Sequential(
            nn.Upsample(scale_factor=2), # (n, 64, 4, 256, 256)
            nn.Conv3d(64, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, kernel_size=1, padding=0, stride=1), # (n, out_channels, 4, 256, 256)
            nn.Sigmoid()
        )

        self.foreground = nn.Sequential(
            nn.Upsample(scale_factor=2), # (n, 64, 4, 256, 256)
            nn.Conv3d(64, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, kernel_size=1, padding=0, stride=1), # (n, out_channels, 4, 256, 256)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return self.mask(x), self.foreground(x)

class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super().__init__()
        self.decoder_bg = Decoder_Background(in_channels=in_channels, out_channels=out_channels)
        self.decoder_fg = Decoder_Foreground(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        b = self.decoder_bg(x)
        m, f = self.decoder_fg(x)

        out = (m * f) + ((1 - m) * b.unsqueeze(2).expand(-1, -1, 4, -1, -1))
        return out, m, f, b

if __name__ == "__main__":
    inputs = torch.randn(1, 1, 4, 256, 256)

    netE = Encoder(in_channels=1)
    latent = netE(inputs)
    print(latent.shape)

    netD = Decoder(in_channels=384, out_channels=1)
    outputs = netD(latent)[0]
    print(outputs.shape)