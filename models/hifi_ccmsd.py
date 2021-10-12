# Class-conditional HiFi-GAN Multi-Scale Discriminator

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, AvgPool1d, Embedding, Linear
from torch.nn.utils import weight_norm, spectral_norm

LRELU_SLOPE = 0.1

class CCDiscriminatorS(torch.nn.Module):
    def __init__(
        self,
        audio_input_dim, 
        n_classes, 
        verbose=False, 
        lrelu_slope=0.1,
        use_spectral_norm=False
    ):
        super(CCDiscriminatorS, self).__init__()
        self.verbose = verbose
        self.lrelu_slope = lrelu_slope
        self.audio_input_dim = audio_input_dim

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
            norm_f(Conv1d(2, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

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

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class CCMultiScaleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        n_classes,
        verbose=False
    ):
        super(CCMultiScaleDiscriminator, self).__init__()
        self.verbose = verbose
        self.audio_input_dim = 8192

        self.discriminators = torch.nn.ModuleList([
            CCDiscriminatorS(self.audio_input_dim, n_classes, self.verbose, use_spectral_norm=True),
            CCDiscriminatorS(int(self.audio_input_dim/2), n_classes, self.verbose),
            CCDiscriminatorS(int(self.audio_input_dim/4), n_classes, self.verbose),
        ])
        self.meanpools = torch.nn.ModuleList([
            AvgPool1d(4, 2, padding=1),
            AvgPool1d(4, 2, padding=1)
        ])

    def forward(
        self, 
        y, 
        y_hat,
        labels_y,
        labels_y_hat
    ):  
        if self.verbose:
            print('-- CCMultiScaleDiscriminator --')
            print('y: ', y.shape)
            print('y_hat: ', y_hat.shape)
            print('labels_y: ', labels_y.shape)
            print('labels_y_hat: ', labels_y_hat.shape)

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y, labels_y)
            y_d_g, fmap_g = d(y_hat, labels_y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs