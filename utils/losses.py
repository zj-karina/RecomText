import torch.nn as nn


def get_losses(name_contrastive_loss):
    if name_contrastive_loss == 'cos_emb':
        contrastive_loss_fn = nn.CosineEmbeddingLoss()
    else:
        pass # for future experiments

    recommendation_loss_fn = nn.CrossEntropyLoss()
    return recommendation_loss_fn, contrastive_loss_fn
