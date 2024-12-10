import torch
from transformers import AutoTokenizer
from utils.mappings import load_mappings

def inference(text, user_id, model, tokenizer, mappings, device):
    """
    Perform inference with the trained model.
    
    Args:
        text (str): Input text
        user_id: User identifier
        model: Trained model
        tokenizer: Tokenizer
        mappings: ID mappings
        device: Torch device
    
    Returns:
        predictions: Model predictions
    """
    model.eval()
    with torch.no_grad():
        # Convert input data
        mapped_user_id = mappings['user_id_map'].get(user_id, mappings['unknown_user_idx'])
        
        # Tokenization
        text_encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        text_encoding = {k: v.to(device) for k, v in text_encoding.items()}
        user_id_tensor = torch.tensor([mapped_user_id], dtype=torch.long).to(device)
        
        # Get embeddings
        items_embeddings, user_embeddings = model(
            text_encoding,
            text_encoding,  # use same text for user_text_inputs
            user_id_tensor,
            user_id_tensor
        )
        
        # Get predictions (example: cosine similarity)
        similarities = torch.nn.functional.cosine_similarity(
            user_embeddings.unsqueeze(1),
            items_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Convert indices back to original IDs
        predictions = [
            mappings['reverse_item_id_map'][idx.item()]
            for idx in torch.topk(similarities[0], k=10).indices
        ]
        
        return predictions

if __name__ == "__main__":
    # Пример использования
    model_path = "path/to/saved/model"
    mappings = load_mappings()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загрузка модели и токенизатора
    model = torch.load(model_path)
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("/storage/kromanova/models/multilingual-e5-large")
    
    # Пример инференса
    text = "example text"
    user_id = "example_user_id"
    
    predictions = inference(text, user_id, model, tokenizer, mappings, device)
    print(f"Top predictions: {predictions}") 