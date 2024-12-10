
import torch
from torch.utils.data import Dataset

def read_news(news_path):
    """Read news data from the given path."""
    news_data = []
    with open(news_path, 'r', encoding='utf-8') as f:
        for line in f:
            news_data.append(line.strip())
    return news_data

def read_behaviors(behaviors_path):
    """Read user behaviors from the given path."""
    behaviors = []
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line in f:
            behaviors.append(line.strip().split('	'))
    return behaviors

class BuildTrainDataset(Dataset):
    """Dataset for training."""
    def __init__(self, news_data, behaviors, tokenizer, max_length=128):
        self.news_data = news_data
        self.behaviors = behaviors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        behavior = self.behaviors[idx]
        user_id, item_id, label = behavior[0], behavior[1], int(behavior[2])
        text = self.news_data[int(item_id)]
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}, label

def eval_model(model, dataloader, criterion):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def get_item_embeddings(model, news_data, tokenizer, device):
    """Generate embeddings for news items."""
    model.eval()
    embeddings = []
    for text in news_data:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            output = model(**inputs)
            embeddings.append(output.last_hidden_state.mean(dim=1))
    return torch.cat(embeddings, dim=0)
