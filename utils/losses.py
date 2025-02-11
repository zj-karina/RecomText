import torch.nn as nn


def get_losses(name_contrastive_loss):
    if name_contrastive_loss == 'infonce':
        contrastive_loss_fn = nn.CrossEntropyLoss() # info_nce_loss_fn
    else:
        contrastive_loss_fn = nn.CosineEmbeddingLoss()

    recommendation_loss_fn = nn.CrossEntropyLoss()
    return recommendation_loss_fn, contrastive_loss_fn
