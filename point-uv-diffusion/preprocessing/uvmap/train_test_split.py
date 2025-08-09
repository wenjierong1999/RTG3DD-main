import pandas as pd
import random
import os

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


def train_test_split_shapenet(flag_csv_path,
                              split_file_output,
                              category,
                              train_ratio = 0.9,
                              seed = 912240):

    random.seed(seed)

    df = pd.read_csv(flag_csv_path)
    print(df.info())
    #print(df['category'].value_counts())
    df_filtered = df[
        (df['is_complete'] == True) &
        (df['is_textured'] == True) &
        (df['category'] == int(category))
    ]
    model_ids = df_filtered['model_id'].tolist()

    random.shuffle(model_ids)
    split_idx = int(len(model_ids) * train_ratio)
    train_ids = model_ids[:split_idx]
    print(f'[{category:}] collected {len(train_ids)} samples for training')
    test_ids = model_ids[split_idx:]
    print(f'[{category:}] collected {len(test_ids)} samples for testing')

    os.makedirs(os.path.join(split_file_output, 'final_split_files', category), exist_ok=True)

    with open(os.path.join(split_file_output, 'final_split_files', category, 'train.lst'), 'w') as f:
        for model_id in train_ids:
            f.write(model_id + '\n')

    with open(os.path.join(split_file_output, 'final_split_files', category, 'test.lst'), 'w') as f:
        for model_id in test_ids:
            f.write(model_id + '\n')

if __name__ == '__main__':

    # train_test_split_future3d(
    #     flag_csv_path='/data/leuven/375/vsc37593/my_py_projects/Future3d-24views-flag.csv',
    #     split_file_output='/scratch/leuven/375/vsc37593/3D-FUTURE-Preprocessed-24Views',
    #     category= 'Sofa'
    # )

    train_test_split_shapenet(
        flag_csv_path='/data/leuven/375/vsc37593/my_py_projects/shapenet-flag.csv',
        split_file_output='/scratch/leuven/375/vsc37593/ShapeNet-Preprocessed',
        category= '4379243'
    )

    train_test_split_shapenet(
        flag_csv_path='/data/leuven/375/vsc37593/my_py_projects/shapenet-flag.csv',
        split_file_output='/scratch/leuven/375/vsc37593/ShapeNet-Preprocessed',
        category= '3001627'
    )