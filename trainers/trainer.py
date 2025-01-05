import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import get_losses
from utils.metrics import compute_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.contrastive_loss_fn, self.recommendation_loss_fn = get_losses()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_recommendation_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            loss, c_loss, r_loss = self.training_step(batch)
            
            total_loss += loss
            total_contrastive_loss += c_loss
            total_recommendation_loss += r_loss

        return {
            'loss': total_loss / len(self.train_loader),
            'contrastive_loss': total_contrastive_loss / len(self.train_loader),
            'recommendation_loss': total_recommendation_loss / len(self.train_loader)
        }

    def training_step(self, batch):
        items_text_inputs, user_text_inputs, item_ids, user_ids = [
            self.to_device(x) for x in batch
        ]

        # Forward pass
        items_embeddings, user_embeddings = self.model(
            items_text_inputs, user_text_inputs, item_ids, user_ids
        )

        # Normalize embeddings
        items_embeddings = F.normalize(items_embeddings, p=2, dim=1)
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

        # Losses
        recommendation_loss = self.compute_recommendation_loss(
            user_embeddings, items_embeddings
        )
        contrastive_loss = self.compute_contrastive_loss(
            items_embeddings, user_embeddings
        )

        # Total loss
        loss = contrastive_loss + self.config['training']['lambda_rec'] * recommendation_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), contrastive_loss.item(), recommendation_loss.item()

    def compute_recommendation_loss(self, user_embeddings, items_embeddings):
        logits = torch.matmul(user_embeddings, items_embeddings.T)
        labels = torch.arange(len(user_embeddings)).to(self.device)
        return self.recommendation_loss_fn(logits, labels)

    def compute_contrastive_loss(self, items_embeddings, user_embeddings):
        batch_size = items_embeddings.size(0)
        positive_labels = torch.ones(batch_size, device=self.device)
        
        contrastive_goods_loss = self.contrastive_loss_fn(
            items_embeddings,
            items_embeddings.roll(shifts=1, dims=0),
            positive_labels
        )
        
        contrastive_users_loss = self.contrastive_loss_fn(
            user_embeddings,
            user_embeddings.roll(shifts=1, dims=0),
            positive_labels
        )
        
        return contrastive_goods_loss + contrastive_users_loss

    def to_device(self, x):
        if isinstance(x, dict):
            return {k: v.to(self.device) for k, v in x.items()}
        return x.to(self.device)