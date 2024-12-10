import pickle
import json
import os

def save_mappings(dataset, path='mappings/'):
    """Save ID mappings to both pickle and JSON formats."""
    os.makedirs(path, exist_ok=True)
    
    # Save using pickle (supports all Python data types)
    mappings = {
        'user_id_map': dataset.user_id_map,
        'item_id_map': dataset.item_id_map,
        'reverse_user_id_map': dataset.reverse_user_id_map,
        'reverse_item_id_map': dataset.reverse_item_id_map
    }
    
    with open(f'{path}id_mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    
    # Save using json (with key conversion to strings)
    json_mappings = {
        'user_id_map': {str(k): v for k, v in dataset.user_id_map.items()},
        'item_id_map': {str(k): v for k, v in dataset.item_id_map.items()},
        'reverse_user_id_map': {str(k): str(v) for k, v in dataset.reverse_user_id_map.items()},
        'reverse_item_id_map': {str(k): str(v) for k, v in dataset.reverse_item_id_map.items()}
    }
    
    with open(f'{path}id_mappings.json', 'w') as f:
        json.dump(json_mappings, f)

def load_mappings(path='mappings/id_mappings.pkl'):
    """Load ID mappings from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f) 