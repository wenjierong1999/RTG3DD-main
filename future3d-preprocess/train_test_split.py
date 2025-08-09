import pandas as pd
import random
import os
import json

def train_test_split_future3d(flag_csv_path,
                              split_file_output,
                              category,
                              train_ratio = 0.9,
                              seed = 912240):

    random.seed(seed)

    df = pd.read_csv(flag_csv_path)
    df_filtered = df[
        (df['is_complete'] == True) &
        (df['category'] == category)
    ]
    model_ids = df_filtered['model_id'].tolist()

    random.shuffle(model_ids)
    split_idx = int(len(model_ids) * train_ratio)
    train_ids = model_ids[:split_idx]
    test_ids = model_ids[split_idx:]

    os.makedirs(os.path.join(split_file_output, 'final_split_files', category), exist_ok=True)

    with open(os.path.join(split_file_output, 'final_split_files', category, 'train.lst'), 'w') as f:
        for model_id in train_ids:
            f.write(model_id + '\n')

    with open(os.path.join(split_file_output, 'final_split_files', category, 'test.lst'), 'w') as f:
        for model_id in test_ids:
            f.write(model_id + '\n')

def train_test_split_future3d_from_json(json_path,
                                        split_file_output,
                                        categories,
                                        train_ratio = 0.9,
                                        seed = 912240):
    '''
    Split models from JSON metadata into train/test sets based on super-category.
    '''
    random.seed(seed)

    with open(json_path, 'r') as f:
        data = json.load(f)
    
        # Group entries by super-category
    for category in categories:
        filtered_models = [
            entry['model_id'] for entry in data
            if entry.get('super-category') == category
        ]
        random.shuffle(filtered_models)
        split_idx = int(len(filtered_models) * train_ratio)
        train_ids = filtered_models[:split_idx]
        test_ids = filtered_models[split_idx:]

        output_dir = os.path.join(split_file_output, 'final_split_files', category)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'train.lst'), 'w') as f:
            for model_id in train_ids:
                f.write(model_id + '\n')
        
        with open(os.path.join(output_dir, 'test.lst'), 'w') as f:
            for model_id in test_ids:
                f.write(model_id + '\n')

if __name__ == '__main__':

    train_test_split_future3d_from_json(
        json_path='model_info.json',
        split_file_output='./',
        categories=['Table'],
        train_ratio=0.9
    )
        
