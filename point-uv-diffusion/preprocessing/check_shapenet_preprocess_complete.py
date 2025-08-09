import pandas as pd
import os
from datetime import datetime
import json

def is_preprocessing_complete(category, model_id, save_dir, uv_resolution = 512, fps_num = 4096):
    """
    Check whether all critical output files for a mesh preprocessing run exist.
    """
    check_list = [
        # Camera
        # os.path.join(save_dir, 'rendered_views', category, 'camera', model_id, 'elevation.npy'),
        # os.path.join(save_dir, 'rendered_views', category, 'camera', model_id, 'rotation.npy'),
        # # At least one rendered view image
        # os.path.join(save_dir, 'rendered_views', category, 'img', model_id, '000.png'),

        # CLIP conditioning image + feature
        os.path.join(save_dir, 'clip_image_data', category, f'{model_id}.png'),
        os.path.join(save_dir, 'clip_image_data', category, f'{model_id}.pt'),
        # Coarse model npz
        os.path.join(save_dir, 'coarse_model', category, f'save_{fps_num}', f'{model_id}.npz'),
        os.path.join(save_dir, 'coarse_model', category, f'save_fpsuv_{fps_num}', f'{model_id}.png'),
        # UV model outputs (6 files minimum)
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_texture_{uv_resolution}.png'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_texture_hom_{uv_resolution}.png'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_mask_{uv_resolution}.png'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_position_{uv_resolution}.npz'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_texture_{uv_resolution}.obj'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_texture_{uv_resolution}.mtl'),
    ]
    return all(os.path.isfile(p) for p in check_list)

def check_if_textured(model_path: str) -> bool:
    """
    Checks if a model_id folder (e.g., '1a00aa6b...') contains an 'images' subfolder,
    indicating it's textured.
    The model_path here refers to the path within the SHAPENET_RAW_DIR.
    """
    images_folder_path = os.path.join(model_path, 'images')
    return os.path.isdir(images_folder_path)


def generate_shapenet_flag_csv(
    shapenet_raw_path,
    shapenet_preprocess_path,
    output_flag_csv_path
):

    # Get all categories and model_ids
    all_shapenet_models = []

    if not os.path.exists(shapenet_raw_path):
        raise FileNotFoundError
    
    for category in os.listdir(shapenet_raw_path):
        category_path = os.path.join(shapenet_raw_path, category)
        if os.path.isdir(category_path):
            # Inside each category directory, assume each folder name is a model_id
            for model_id in os.listdir(category_path):
                model_id_path = os.path.join(category_path, model_id)
                if os.path.isdir(model_id_path):  # Ensure it's a directory
                    all_shapenet_models.append({'category': category, 
                                                'model_id': model_id,
                                                'raw_model_path': model_id_path
                                                })
    
        # Create an empty DataFrame
    df = pd.DataFrame({
        'category': pd.Series(dtype='str'),
        'model_id': pd.Series(dtype='str'),
        'is_complete': pd.Series(dtype='bool'),
        'is_textured': pd.Series(dtype='bool'),
        'timestamp': pd.Series(dtype='str'),
    })

    # Populate the DataFrame
    for model_info in all_shapenet_models:
        category = model_info['category']
        model_id = model_info['model_id']
        raw_model_path = model_info['raw_model_path']

        # Check if the model has completed preprocessing
        is_complete = is_preprocessing_complete(category=category, model_id=model_id, save_dir=shapenet_preprocess_path)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S') if is_complete else ''

        # check if model is textured
        is_textured = check_if_textured(raw_model_path)

        new_row = pd.DataFrame([{
            'category': category,
            'model_id': model_id,
            'is_complete': is_complete,
            'is_textured': is_textured,
            'timestamp': timestamp
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(output_flag_csv_path, index=False)


if __name__ == "__main__":

    generate_shapenet_flag_csv(
        shapenet_preprocess_path='/scratch/leuven/375/vsc37593/ShapeNet-Preprocessed',
        shapenet_raw_path='/scratch/leuven/375/vsc37593/ShapeNet_raw',
        output_flag_csv_path='/data/leuven/375/vsc37593/my_py_projects/shapenet-flag.csv'
    )

