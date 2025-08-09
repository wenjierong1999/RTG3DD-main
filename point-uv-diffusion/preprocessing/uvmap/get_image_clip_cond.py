import os
import torch
import clip
import PIL
from pathlib import Path
import shutil
import logging


'''
Demo code of CLIP library
from https://github.com/openai/CLIP
'''

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-L/14@336px", device=device)

# image = preprocess(Image.open("/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-draft/preprocessing/get_image_clip_cond/CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


def extract_and_save_clip_embedding(data_dir,
                                    save_dir,
                                    mesh_id,
                                    category,
                                    clip_model_name = 'ViT-L/14',
                                    device = 'cuda' if torch.cuda.is_available() else 'cpu'):

    # save_dir is the main saving folder!!!
    os.makedirs(os.path.join(save_dir, 'clip_image_data',category), exist_ok=True)
    clip_model, preprocess = clip.load(clip_model_name, device=device)

    # open and preprocess image
    img_path = os.path.join(data_dir, mesh_id, 'image.jpg')
    img = PIL.Image.open(img_path).convert("RGB")
    img_input = preprocess(img).unsqueeze(0).to(device)

    # extract clip embedding
    with torch.no_grad():
        image_features = clip_model.encode_image(img_input)
        #print(image_features.shape)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # saving
    save_img_path = os.path.join(save_dir, 'clip_image_data', category, "%s.png" % mesh_id)
    save_pt_path = os.path.join(save_dir, 'clip_image_data', category, "%s.pt" % mesh_id)
    img.save(save_img_path, format='PNG')
    torch.save(image_features.cpu(), save_pt_path)
    logging.info('Clip feature extraction finished.')

def generate_clip_data():

    raise NotImplementedError




