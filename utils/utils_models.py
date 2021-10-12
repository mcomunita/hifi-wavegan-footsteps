import torch.nn as nn

# same initialisation as in tensorflow for consistency with original wavegan
def weights_init(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, a=-0.05, b=0.05) # default in tf
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0) # default in tf
        nn.init.constant_(m.bias, 0) # default in tf
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1) # default in tf in tf (weight = gamma)
        nn.init.constant_(m.bias, 0) # default in tf (bias = beta)
        nn.init.constant_(m.running_mean, 0) # default in tf
        nn.init.constant_(m.running_var, 1) # default in tf
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, gain=1.0) # default in tf
        nn.init.constant_(m.bias, 0) # default in tf
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(m.weight, gain=1.0) # default in tf
        nn.init.constant_(m.bias, 0) # default in tf

# HIFI-Gan utils

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)