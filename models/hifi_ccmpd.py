# Class-conditional HiFi-GAN Multi-Period Discriminator

import torch
import torch.nn.functional as F
from torch.nn import Conv2d, Embedding, Linear
from torch.nn.utils import weight_norm, spectral_norm
from utils import get_padding

class CCDiscriminatorP(torch.nn.Module):
    def __init__(
        self, 
        period,
        audio_input_dim,
        n_classes, 
        kernel_size=5, 
        stride=3,
        lrelu_slope=0.1,
        use_spectral_norm=False,
        verbose=False
    ):   
        super(CCDiscriminatorP, self).__init__()
        self.period = period
        self.audio_input_dim = audio_input_dim
        self.lrelu_slope = lrelu_slope
        self.verbose = verbose

        self.label_emb1 = Embedding(
            num_embeddings=n_classes, 
            embedding_dim=n_classes * 20, 
        )
        self.label_emb2 = Linear(
            in_features=n_classes * 20,
            out_features=audio_input_dim
        )
        # emb.shape = [batch_size, audio_input_dim]
        # reshape to emb.shape = [batch_size, 1, audio_input_dim]

        # x.shape = [batch_size, 1, audio_input_dim]
        
        # [x, emb]
        # x.shape = [x, emb]
        # x.shape = [batch_size, 2, audio_input_dim]
        
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch.nn.ModuleList([
            norm_f(Conv2d(2, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(
        self, 
        x, 
        labels
    ):
        fmap = []

        # get label embedding
        emb = self.label_emb2(self.label_emb1(labels))
        emb = emb.view(-1, 1, self.audio_input_dim)
        if self.verbose:
            print('emb shape: ', emb.shape)
            print('x shape: ', x.shape)
        # concat input and label
        x = torch.cat([x, emb], dim=1)
        if self.verbose:
            print('x+label shape: ', x.shape)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class CCMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(
        self,
        architecture_size,
        n_classes,
        verbose=False
    ):
        super(CCMultiPeriodDiscriminator, self).__init__()
        self.verbose = verbose

        if architecture_size == 'large':
            self.audio_input_dim = 65536
        elif architecture_size == 'medium':
            self.audio_input_dim = 32768
        elif architecture_size == 'small':
            self.audio_input_dim = 16384
        elif architecture_size == 'extrasmall':
            self.audio_input_dim = 8192
            
        self.discriminators = torch.nn.ModuleList([
            CCDiscriminatorP(period=2, audio_input_dim=self.audio_input_dim, n_classes=n_classes, verbose=self.verbose),
            CCDiscriminatorP(period=3, audio_input_dim=self.audio_input_dim, n_classes=n_classes, verbose=self.verbose),
            CCDiscriminatorP(period=5, audio_input_dim=self.audio_input_dim, n_classes=n_classes, verbose=self.verbose),
            CCDiscriminatorP(period=7, audio_input_dim=self.audio_input_dim, n_classes=n_classes, verbose=self.verbose),
            CCDiscriminatorP(period=11, audio_input_dim=self.audio_input_dim, n_classes=n_classes, verbose=self.verbose),
        ])

    def forward(
        self, 
        y, 
        y_hat, 
        labels_y, 
        labels_y_hat
    ):
        if self.verbose:
            print('-- CCMultiPeriodDiscriminator --')
            print('y: ', y.shape)
            print('y_hat: ', y_hat.shape)
            print('labels_y: ', labels_y.shape)
            print('labels_y_hat: ', labels_y_hat.shape)

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y, labels_y)
            y_d_g, fmap_g = d(y_hat, labels_y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
            if self. verbose:
                print(f'disc {i} y_d_r: ', y_d_r.shape)
                print(f'disc {i} y_d_g: ', y_d_g.shape)
                print(f'disc {i} fmap_r[0]: ', fmap_r[0].shape)
                print(f'disc {i} fmap_g[0]: ', fmap_g[0].shape)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs