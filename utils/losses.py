import torch.nn as nn

def get_losses():
    contrastive_loss_fn = nn.CosineEmbeddingLoss()
    recommendation_loss_fn = nn.CrossEntropyLoss()
    return contrastive_loss_fn, recommendation_loss_fn