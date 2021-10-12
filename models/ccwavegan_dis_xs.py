# Class-conditional WaveGAN Discriminator
# xs = extrasmall = 8192 samples
"""
Weights initialisation and layers parameters explicitly set to 
tesorflow's default values for consistency with original WaveGAN.
"""

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, Embedding, Linear
from utils.utils_models import weights_init

# ================ DISCRIMINATOR ================ #
class CCWaveGANDiscriminator(torch.nn.Module):
    
    def __init__(
        self,
        alpha_lrelu,
        n_classes,
        verbose
    ):
        super(CCWaveGANDiscriminator, self).__init__()
        
        self.alpha_lrelu = alpha_lrelu
        self.n_classes = n_classes
        self.verbose = verbose
        self.audio_input_dim = 8192
        
        self.label_emb1 =   Embedding(
                                num_embeddings=n_classes, 
                                embedding_dim=n_classes * 20,
                                dtype=torch.float32
                            )
        self.label_emb2 =   Linear(
                                n_classes * 20, 
                                self.audio_input_dim,
                                dtype=torch.float32
                            )
        # emb.shape = [batch_size, audio_input_dim]
        # reshape to emb.shape = [batch_size, 1, audio_input_dim]

        # x.shape = [batch_size, 1, audio_input_dim]
        
        # [x, emb]
        # x.shape = [x, emb]
        # x.shape = [batch_size, 2, audio_input_dim]

        # Layer 0
        self.conv0 = Conv1d(2, 64, kernel_size=25, stride=2, padding=11, dilation=1, padding_mode='zeros', dtype=torch.float32)
        
        # Layer 1
        self.conv1 = Conv1d(64, 128, kernel_size=25, stride=4, padding=11, dilation=1, padding_mode='zeros', dtype=torch.float32)

        # Layer 2
        self.conv2 = Conv1d(128, 256, kernel_size=25, stride=4, padding=11, dilation=1, padding_mode='zeros', dtype=torch.float32)

        # Layer 3
        self.conv3 = Conv1d(256, 512, kernel_size=25, stride=4, padding=11, dilation=1, padding_mode='zeros', dtype=torch.float32)

        # Layer 4
        self.conv4 = Conv1d(512, 1024, kernel_size=25, stride=4, padding=11, dilation=1, padding_mode='zeros', dtype=torch.float32)

        # flatten to x.shape = [batch_size, fc_input_size]

        self.fc1 = Linear(self.audio_input_dim, 1, dtype=torch.float32)

        self.apply(weights_init)  

    def forward(self, x, labels):
        # Embedding
        emb = self.label_emb2(self.label_emb1(labels))
        emb = emb.view(-1, 1, self.audio_input_dim)
        if self.verbose:
            print('dis embedding shape: ', emb.shape)
            print('dis input shape: ', x.shape)
        
        # Concat
        x = torch.cat([x, emb], dim=1)

        if self.verbose:
            print('dis input+label shape: ', x.shape)

        # pass through conv layers
        x = F.leaky_relu(self.conv0(x), self.alpha_lrelu)
        if self.verbose:
            print('dis conv0 out shape: ', x.shape)

        x = F.leaky_relu(self.conv1(x), self.alpha_lrelu)
        if self.verbose:
            print('dis conv1 out shape: ', x.shape)
            
        x = F.leaky_relu(self.conv2(x), self.alpha_lrelu)
        if self.verbose:
            print('dis conv2 out shape: ', x.shape)
            
        x = F.leaky_relu(self.conv3(x), self.alpha_lrelu)
        if self.verbose:
            print('dis conv3 out shape: ', x.shape)
            
        x = F.leaky_relu(self.conv4(x), self.alpha_lrelu)
        if self.verbose:
            print('dis conv4 out shape: ', x.shape)
        
        x = x.view(-1, self.audio_input_dim)
        x = self.fc1(x)
        if self.verbose:
            print('dis output shape: ', x.shape)

        return x