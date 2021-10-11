# WGANGP Utils

def gradients_status(model, flag):
    for p in model.parameters():
        p.requires_grad = flag