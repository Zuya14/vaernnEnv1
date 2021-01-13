import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
from typing import Any

class BasicConv(jit.ScriptModule):

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        self.fc = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.fc(x)

class BasicConvT(jit.ScriptModule):

    def __init__(self, in_channels: int, out_channels: int, lastFlag: bool=False, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        self.fc = nn.ReLU(inplace=True)
        self._forward = self._forward2 if lastFlag else self._forward1

    def _forward1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.fc(x)

    def _forward2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

class InceptionModule(jit.ScriptModule):

    def __init__(self, in_channels: int, out_channels: int, red_channels: int) -> None:
        super().__init__()

        self.conv1 = BasicConv(in_channels, out_channels, kernel_size=1)

        self.conv3_red = BasicConv(in_channels, red_channels, kernel_size=1)
        self.conv3 = BasicConv(red_channels, out_channels, kernel_size=3, padding=1)

        self.conv5_red = BasicConv(in_channels, red_channels, kernel_size=1)
        self.conv3_1 = BasicConv(red_channels, out_channels, kernel_size=3, padding=1)
        self.conv3_2 = BasicConv(out_channels, out_channels, kernel_size=3, padding=1)

        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv1 = BasicConv(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y_c1 = self.conv1(x)

        y_c3 = self.conv3_red(x)
        y_c3 = self.conv3(y_c3)

        y_c5 = self.conv5_red(x)
        y_c5 = self.conv3_1(y_c5)
        y_c5 = self.conv3_2(y_c5)

        y_p3 = self.pool3(x)
        y_p3 = self.pool_conv1(y_p3)

        out = y_c1 + y_c3 + y_c5 + y_p3

        return out

class InceptionEncoder(jit.ScriptModule):

    def __init__(self, in_channels: int, out_channels: int, red_channels: int, stride: int) -> None:
        super().__init__()

        self.conv1 = BasicConv(in_channels, out_channels, kernel_size=1, stride=stride)

        self.conv3_red = BasicConv(in_channels, red_channels, kernel_size=1)
        self.conv3 = BasicConv(red_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.conv5_red = BasicConv(in_channels, red_channels, kernel_size=1)
        self.conv3_1 = BasicConv(red_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3_2 = BasicConv(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y_c1 = self.conv1(x)

        y_c3 = self.conv3_red(x)
        y_c3 = self.conv3(y_c3)

        y_c5 = self.conv5_red(x)
        y_c5 = self.conv3_1(y_c5)
        y_c5 = self.conv3_2(y_c5)
        
        out = y_c1 + y_c3 + y_c5

        return out

class InceptionDecoder(jit.ScriptModule):

    def __init__(self, in_channels: int, out_channels: int, red_channels: int, stride: int, lastFlag: bool=False) -> None:
        super().__init__()

        padding = 1 if stride==2 else 2

        self.conv1 = BasicConvT(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=padding, lastFlag=lastFlag)

        self.conv3_red = BasicConvT(in_channels, red_channels, kernel_size=1)
        self.conv3 = BasicConvT(red_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=padding, lastFlag=lastFlag)

        self.conv5_red = BasicConv(in_channels, red_channels, kernel_size=1)
        self.conv3_1 = BasicConvT(red_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=padding)
        self.conv3_2 = BasicConvT(out_channels, out_channels, kernel_size=3, padding=1, lastFlag=lastFlag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y_c1 = self.conv1(x)

        y_c3 = self.conv3_red(x)
        y_c3 = self.conv3(y_c3)

        y_c5 = self.conv5_red(x)
        y_c5 = self.conv3_1(y_c5)
        y_c5 = self.conv3_2(y_c5)

        out = y_c1 + y_c3 + y_c5

        return out

class Encoder(jit.ScriptModule):

    def __init__(self, first_channel: int, latent_size: int, red_times: int, repeat: int=0, channel_inc: int=2):
        super().__init__()

        k = first_channel

        self.repeat = repeat

        c = channel_inc

        self.conv1 = InceptionEncoder(       1,        k,        k//red_times, stride=3)
        self.conv2 = InceptionEncoder(       k,      c*k,      c*k//red_times, stride=3)
        self.conv3 = InceptionEncoder(     c*k, (c**2)*k, (c**2)*k//red_times, stride=3)
        self.conv4 = InceptionEncoder((c**2)*k, (c**3)*k, (c**3)*k//red_times, stride=2)
        self.conv5 = InceptionEncoder((c**3)*k, (c**4)*k, (c**4)*k//red_times, stride=2)
        self.conv6 = InceptionEncoder((c**4)*k, (c**5)*k, (c**5)*k//red_times, stride=2)
        
        if self.repeat > 0:
            self.module1 = nn.Sequential(*[InceptionModule(       k,        k,        k//red_times) for _ in range(self.repeat)])
            self.module2 = nn.Sequential(*[InceptionModule(     c*k,      c*k,      c*k//red_times) for _ in range(self.repeat)])
            self.module3 = nn.Sequential(*[InceptionModule((c**2)*k, (c**2)*k, (c**2)*k//red_times) for _ in range(self.repeat)])
            self.module4 = nn.Sequential(*[InceptionModule((c**3)*k, (c**3)*k, (c**3)*k//red_times) for _ in range(self.repeat)])
            self.module5 = nn.Sequential(*[InceptionModule((c**4)*k, (c**4)*k, (c**4)*k//red_times) for _ in range(self.repeat)])
            self.module6 = nn.Sequential(*[InceptionModule((c**5)*k, (c**5)*k, (c**5)*k//red_times) for _ in range(self.repeat)])

        self._forward = self._forward2 if self.repeat > 0 else self._forward1

        self.embedding_size = 5*(c**5)*k

        self.fc1 = nn.Linear(self.embedding_size, latent_size)
        self.fc2 = nn.Linear(self.embedding_size, latent_size)

    def _forward1(self, x: torch.Tensor) -> torch.Tensor:

        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)

        return h

    def _forward2(self, x: torch.Tensor) -> torch.Tensor:

        h = self.conv1(x)
        h = self.module1(h)
        h = self.conv2(h)
        h = self.module2(h)
        h = self.conv3(h)
        h = self.module3(h)
        h = self.conv4(h)
        h = self.module4(h)
        h = self.conv5(h)
        h = self.module5(h)
        h = self.conv6(h)
        h = self.module6(h)

        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self._forward(x) 

        h = h.view(-1, self.embedding_size)

        mu     = self.fc1(h)
        logvar = self.fc2(h)

        return mu, logvar

class Decoder(jit.ScriptModule):

    def __init__(self, last_channel: int, latent_size: int, red_times: int, repeat: int=0, channel_inc: int=2):
        super().__init__()

        k = last_channel
        self.repeat = repeat

        c = channel_inc

        self.embedding_size = 5*(c**5)*k

        self.fc = nn.Linear(latent_size, self.embedding_size)

        self.convT1 = InceptionDecoder((c**5)*k, (c**4)*k, (c**5)*k//red_times, stride=2)
        self.convT2 = InceptionDecoder((c**4)*k, (c**3)*k, (c**4)*k//red_times, stride=2)
        self.convT3 = InceptionDecoder((c**3)*k, (c**2)*k, (c**3)*k//red_times, stride=2)
        self.convT4 = InceptionDecoder((c**2)*k,      c*k, (c**2)*k//red_times, stride=3)
        self.convT5 = InceptionDecoder(     c*k,        k,      c*k//red_times, stride=3)
        self.convT6 = InceptionDecoder(       k,        1,        k//red_times, stride=3, lastFlag=True)

        if self.repeat > 0:
            self.module1 = nn.Sequential(*[InceptionModule((c**5)*k, (c**5)*k, (c**5)*k//red_times) for _ in range(self.repeat)])
            self.module2 = nn.Sequential(*[InceptionModule((c**4)*k, (c**4)*k, (c**4)*k//red_times) for _ in range(self.repeat)])
            self.module3 = nn.Sequential(*[InceptionModule((c**3)*k, (c**3)*k, (c**3)*k//red_times) for _ in range(self.repeat)])
            self.module4 = nn.Sequential(*[InceptionModule((c**2)*k, (c**2)*k, (c**2)*k//red_times) for _ in range(self.repeat)])
            self.module5 = nn.Sequential(*[InceptionModule(     c*k,      c*k,      c*k//red_times) for _ in range(self.repeat)])
            self.module6 = nn.Sequential(*[InceptionModule(       k,        k,        k//red_times) for _ in range(self.repeat)])

        self._forward = self._forward2 if self.repeat > 0 else self._forward1

        self.sigmoid = nn.Sigmoid()

    def _forward1(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)
        x = self.convT5(x)
        x = self.convT6(x)

        return x

    def _forward2(self, x: torch.Tensor) -> torch.Tensor:

        x = self.module1(x)
        x = self.convT1(x)
        x = self.module2(x)
        x = self.convT2(x)
        x = self.module3(x)
        x = self.convT3(x)
        x = self.module4(x)
        x = self.convT4(x)
        x = self.module5(x)
        x = self.convT5(x)
        x = self.module6(x)
        x = self.convT6(x)

        return x

    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, self.embedding_size//5, 5) # 畳み込み後の最終データサイズが5

        x = self._forward(x) 

        return self.sigmoid(x)

class VAE(jit.ScriptModule):

    def __init__(self, first_channel: int, latent_size: int, red_times: int, repeat: int=0, channel_inc: int=2):
        super().__init__()
        self.encoder = Encoder(first_channel, latent_size, red_times, repeat, channel_inc)
        self.decoder = Decoder(first_channel, latent_size, red_times, repeat, channel_inc)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        sigma = logvar.mul(0.5).exp()
        eps   = torch.randn_like(sigma)

        if self.training:
            z = eps.mul(sigma).add_(mu) # mu + sigma^(1/2) * eps
        else:
            z = mu

        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):

        x2 = torch.clamp(x.view(recon_x.size()), 1e-24, 1.0-1e-24)
        recon_x2 = torch.clamp(recon_x, 1e-24, 1.0-1e-24)

        batch_size = x2.size()[0]

        # print(torch.min(recon_x2))

        # assert (torch.min(recon_x) >= 0. and torch.max(recon_x) <= 1.)
        # assert (torch.min(x2) >= 0. and torch.max(x2) <= 1.)
        # assert (torch.min(recon_x2) >= 0.)
        # assert (torch.max(recon_x2) <= 1.)
        # assert (torch.min(x2) >= 0.)
        # assert (torch.max(x2) <= 1.)

        # BCE = F.binary_cross_entropy(recon_x2, x2, reduction='sum')
        BCE = F.binary_cross_entropy(recon_x2, x2, reduction='sum')/batch_size

        # 0.5*(1 + log(sigma^2) - mu^2 - sigma^2) 
        # 実装ではsigmaがマイナスになるとlogsigmaを求められないためか、2*logsigmaをlogvarと置いて
        # KL距離を0.5*(mu^2 + exp(logvar) −logvar − 1) とする記述が主流?
        # https://qiita.com/nishiha/items/2264da933504fbe3fc68

        # KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)/batch_size
        # KLD = 0
        # print(KLD)
        return BCE, KLD


if __name__ == '__main__':

    x = torch.randn(10,1,1080)

    vae = VAE(first_channel=8, latent_size=18, red_times=1, repeat=1)
    recon_x, mu, logvar = vae(x)

    print(recon_x.size(), mu.size(), logvar.size())