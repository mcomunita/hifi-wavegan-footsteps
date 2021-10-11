# Class-conditional WaveGAN Generator
# xs = extrasmall = 8192 samples
"""
Weights initialisation and layers parameters explicitly set to 
tesorflow's default values for consistency with original WaveGAN.
"""

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, Embedding, Linear, BatchNorm1d, Sequential
from utils.utils_models import weights_init

# custom 1D upsampling convolutional layer
class ConvTranspose1DWithUpsample(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        padding_mode='zeros',
        upsample_mode='zeros',
        dtype=torch.float32
    ):
        super(ConvTranspose1DWithUpsample, self).__init__()
        
        assert upsample_mode in ['zeros', 'nn', 'linear', 'cubic']
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if upsample_mode == 'zeros':
            self.upsample_flag = False
        else:
            self.upsample_flag = True
            self.upsample_rate = stride
            self.stride = 1
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.upsample_mode = upsample_mode
        self.dtype = dtype

        if upsample_mode == 'zeros':
            self.conv = ConvTranspose1d(
                            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding, output_padding=output_padding, 
                            dilation=dilation, padding_mode=padding_mode, dtype=dtype
                        )
        else:
            # upsample w/ interpolation, then convolution
            self.conv = Conv1d(
                            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                            stride=self.stride, padding=padding, 
                            dilation=dilation, padding_mode=padding_mode, dtype=dtype
                        )

    def forward(self, x):
        if self.upsample_flag:
            if self.upsample_mode == 'zeros':
                pass
            elif self.upsample_mode == 'nn':
                _, channels, width = x.shape
                x = torch.unsqueeze(x, dim=1)
                x = torch.nn.functional.interpolate(x, size=[channels, width * self.upsample_rate], mode="nearest")
                x = torch.squeeze(x, dim=1)
            elif self.upsample_mode == 'linear':
                _, channels, width = x.shape
                x = torch.unsqueeze(x, dim=1)
                x = torch.nn.functional.interpolate(x, size=[channels, width * self.upsample_rate], mode="bilinear")
                x = torch.squeeze(x, dim=1)
            elif self.upsample_mode == 'cubic':
                _, channels, width = x.shape
                x = torch.unsqueeze(x, dim=1)
                x = torch.nn.functional.interpolate(x, size=[channels, width * self.upsample_rate], mode="bicubic")
                x = torch.squeeze(x, dim=1)
            else:
                raise NotImplementedError
        x = self.conv(x)
        return x

# ================ GENERATOR ================ #
class CCWaveGANGenerator(torch.nn.Module):
    
    def __init__(
        self,
        latent_dim,
        n_classes,
        upsample_mode,
        verbose
    ):
        super(CCWaveGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.upsample_mode = upsample_mode
        self.verbose = verbose
        self.audio_dim = 8192
        
        # Embedding Layers
        self.label_emb1 =   Embedding(
                                num_embeddings=n_classes, 
                                embedding_dim=n_classes * 20,
                                dtype=torch.float32
                            )
        self.label_emb2 =   Linear(
                                in_features=n_classes * 20, 
                                out_features=8,
                                dtype=torch.float32
                            )
        
        # Reshape from [batch_size, 8] to [batch_size, 1, 8]

        # Input Layer - [batch_size, latent_dim]
        self.fc1 =  Linear(
                        in_features=latent_dim, 
                        out_features=self.audio_dim,
                        dtype=torch.float32
                    )
        
        # Reshape from [batch_size, 8192] to [batch_size, 1024, 8]
        self.bn1 =  BatchNorm1d(
                        num_features=1024, 
                        momentum=0.99, 
                        eps=0.001, 
                        affine=True, 
                        track_running_stats=True,
                        dtype=torch.float32
                    )
        
        # relu(x)
        # x.shape = [batch_size, 1024, 8]
        # x = [x, emb]
        # x.shape = [batch_size, 1024, 8]

        # Layer 0
        self.conv0 =    Sequential(
                            ConvTranspose1DWithUpsample(
                                in_channels=1024+1, out_channels=512, kernel_size=25, 
                                stride=4, padding=11, output_padding=1, 
                                dilation=1, padding_mode='zeros',
                                upsample_mode=upsample_mode, dtype=torch.float32
                            ),
                            BatchNorm1d(512, 0.99, 0.001, True, True, dtype=torch.float32)
                        )

        # Layer 1
        self.conv1 =    Sequential(
                            ConvTranspose1DWithUpsample(512, 256, 25, 4, 11, 1, 1, 'zeros', upsample_mode, dtype=torch.float32),
                            BatchNorm1d(256, 0.99, 0.001, True, True, dtype=torch.float32)
                        )

        # Layer 2
        self.conv2 =    Sequential(
                            ConvTranspose1DWithUpsample(256, 128, 25, 4, 11, 1, 1, 'zeros', upsample_mode, dtype=torch.float32),
                            BatchNorm1d(128, 0.99, 0.001, True, True, dtype=torch.float32)
                        )

        # Layer 3
        self.conv3 =    Sequential(
                            ConvTranspose1DWithUpsample(128, 64, 25, 4, 11, 1, 1, 'zeros', upsample_mode, dtype=torch.float32),
                            BatchNorm1d(64, 0.99, 0.001, True, True, dtype=torch.float32)
                        )

        # Layer 4
        self.conv4 =    Sequential(
                            ConvTranspose1DWithUpsample(64, 1, 25, 4, 11, 1, 1, 'zeros', upsample_mode, dtype=torch.float32)
                        )
        
        self.apply(weights_init)
             
    def forward(self, x, labels):
        # Embedding
        emb = self.label_emb2(self.label_emb1(labels))
        emb = emb.view(-1, 1, 8)
        if self.verbose:
            print('gen embedding shape: ', emb.shape)

        # Input
        x = self.fc1(x)
        x = x.view(-1, 1024, 8)
        x = self.bn1(x)
        x = F.relu(x)
        
        if self.verbose:
            print('gen input shape: ', x.shape)
            
        # Concat
        x = torch.cat([x, emb], dim=1)

        if self.verbose:
            print('gen input+label shape: ', x.shape)
            
        # Convs
        x = F.relu(self.conv0(x))
        if self.verbose:
            print('gen conv0 out shape: ', x.shape)
        x = F.relu(self.conv1(x))
        if self.verbose:
            print('gen conv1 out shape: ', x.shape)
        x = F.relu(self.conv2(x))
        if self.verbose:
            print('gen conv2 out shape: ', x.shape)
        x = F.relu(self.conv3(x))
        if self.verbose:
            print('gen conv3 out shape: ', x.shape)
        x = torch.tanh(self.conv4(x))
        if self.verbose:
            print('gen conv4 out shape: ', x.shape)
        if self.verbose:
            print('gen output shape: ', x.shape)
            
        return x