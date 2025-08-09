import os
import argparse
import numpy as np
import logging
from multiprocessing import Pool
import torch
import shutil
from shutil import copyfile
import time


parser = argparse.ArgumentParser('data_process', add_help=False)

parser.add_argument(
    '--save_folder', type=str, default='./tmp',
    help='path for saving rendered image')
parser.add_argument(
    '--dataset_folder', type=str, default='/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-draft/3D-Future-Demo',
    help='path for downloaded 3d dataset folder')
parser.add_argument(
    '--blender_root', type=str, default='/data/leuven/375/vsc37593/blender-2.90.0-linux64/blender',
    help='path for blender')
parser.add_argument('--model_filename', type=str, default='raw_model.obj', help='file name of obj file')
parser.add_argument('--obj_scale', type=float, default=0.7)
parser.add_argument('--view_num', default=24, type=int)
parser.add_argument('--resolution', default=512, type=int)
parser.add_argument(
    '--synset', type=str, default='03001627',
    help='the category for render')
parser.add_argument('--num_workers', type=int, default=4,
    help='Number of parallel processes (multiprocessing pool)')


args = parser.parse_args()

save_folder = args.save_folder
dataset_folder = args.dataset_folder
blender_root = args.blender_root
model_filename = args.model_filename
synset = args.synset
obj_scale = args.obj_scale
view_num = args.view_num
resolution = args.resolution
num_workers = args.num_workers


'''
/data/leuven/375/vsc37593/blender-2.90.0-linux64/blender \
  -b \
  -P blender_render_multiviews.py \
  -- \
  --output ./tmp \
  /data/leuven/375/vsc37593/my_py_projects/ShapeNet_toy/02828884/1ac6a3d5c76c8b96edccc47bf0dcf5d3/models/model_normalized.obj \
  --scale 0.7 \
  --views 24 \
  --resolution 512
'''

def run_render_cmd(file):
    render_cmd = '%s -b -P blender_render_multiviews.py -- --output %s --synset %s %s  --scale %f --views %d --resolution %d' % (
        blender_root, 
        save_folder, 
        synset,
        os.path.join(dataset_folder, synset, file, 'models',model_filename), 
        obj_scale, 
        view_num, 
        resolution
    )
    os.system(render_cmd)

def render_from_mesh(model_id):
    # print('-> Run rendering.')
    # try:
    #     t0 = time.time()
    #     run_render_cmd(file)
    #     t1 = time.time()
    #     logging.info('It takes %.4f seconds to render %s' % (t1 - t0, file))
    # except Exception as e:
    #     logging.warning(e)

    model_path = os.path.join(dataset_folder, synset, model_id, 'models', model_filename)

    if not os.path.isfile(model_path):
        logging.warning(f"Skipping {model_id} (missing file: {model_filename})")
        return

    cmd = '%s -b -P blender_render_multiviews.py -- --output %s --synset %s %s --scale %f --views %d --resolution %d' % (
        blender_root,
        save_folder,
        synset,
        model_path,
        obj_scale,
        view_num,
        resolution
    )

    try:
        t0 = time.time()
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.time()
        logging.info(f"[DONE] {model_id} in {round(t1 - t0, 2)}s")
    except Exception as e:
        logging.warning(f"[FAIL] {model_id}: {e}")

def generate_dataset(filename):

    synset_path = os.path.join(dataset_folder, synset)
    if not os.path.isdir(synset_path):
        raise ValueError(f"Synset folder not found: {synset_path}")
    
    model_ids = sorted([
        d for d in os.listdir(synset_path)
        if os.path.isdir(os.path.join(synset_path, d))
    ])
    logging.info(f"Found {len(model_ids)} models in synset {synset}. Start rendering...")

    with Pool(processes=num_workers) as pool:
        pool.map(run_render_cmd, model_ids)


def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


if __name__ == '__main__':
    #workers = args.worker
    set_logging(log_path='./log/render_log.log')
    #generate_dataset('1ac6a3d5c76c8b96edccc47bf0dcf5d3')
    generate_dataset(synset)