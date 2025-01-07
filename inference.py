import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from utils.mappings import load_mappings
from tqdm import tqdm
from data.dataset import BuildTrainDataset
from torch.utils.data import DataLoader

def evaluate_predictions(model, val_loader, device, k=10, num_examples=5):
    """
    Evaluate model predictions on validation set and show examples.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Torch device
        k: Number of recommendations to show
        num_examples: Number of user examples to show
    """
    model.eval()
    
    with torch.no_grad():
        # Возьмем один батч для примера
        for batch in val_loader:
            items_text_inputs, user_text_inputs, item_ids, user_ids, categories = [
                x.to(device) for x in batch
            ]

            # Forward pass
            items_embeddings, user_embeddings = model(
                items_text_inputs, user_text_inputs, item_ids, user_ids
            )

            # Нормализация эмбеддингов
            items_embeddings = F.normalize(items_embeddings, p=2, dim=1)
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

            # Вычисляем схожесть для каждого пользователя
            predictions = []
            for user_emb in user_embeddings[:num_examples]:  # Берем только num_examples пользователей
                user_emb = user_emb.unsqueeze(0)
                user_predictions = F.cosine_similarity(
                    user_emb.unsqueeze(0),
                    items_embeddings.unsqueeze(0),
                    dim=2
                )
                predictions.append(user_predictions.squeeze(0))
            
            predictions = torch.stack(predictions)  # [num_examples, n_items]
            
            # Получаем топ-k предсказаний для каждого пользователя
            top_k_scores, top_k_indices = torch.topk(predictions, k=min(k, items_embeddings.size(0)))
            
            # Выводим результаты
            print("\nPrediction Examples:")
            for user_idx in range(min(num_examples, len(user_ids))):
                print(f"\nUser {user_idx + 1}:")
                print(f"User ID: {user_ids[user_idx].item()}")
                print(f"User Text Input: {user_text_inputs['input_ids'][user_idx].tolist()}")  # Нужно декодировать через токенизатор
                print("\nTop recommendations:")
                
                for rank, (score, idx) in enumerate(zip(top_k_scores[user_idx], top_k_indices[user_idx]), 1):
                    print(f"{rank}. Item ID: {item_ids[idx].item()}")
                    print(f"   Score: {score:.4f}")
                    print(f"   Category: {categories[idx].item()}")
                    print(f"   Text Input: {items_text_inputs['input_ids'][idx].tolist()}")  # Нужно декодировать через токенизатор
                
            break  # Выходим после первого батча

if __name__ == "__main__":
    # Загрузка необходимых компонентов
    model_path = "path/to/your/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загрузка модели
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    
    # Создание валидационного датасета и лоадера
    val_dataset = BuildTrainDataset(
        # Добавьте ваши параметры для датасета
        split='val'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Оценка предсказаний
    evaluate_predictions(model, val_loader, device, k=10, num_examples=5) 