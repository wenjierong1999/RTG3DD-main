import math
import torch
import os
import PIL
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, resize_pos_embed, resample_abs_pos_embed
from torch import Tensor
from torchvision.transforms import functional as TVF

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

MODEL_URLS = {
    'vit_base_patch16_224_mae': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
    'vit_small_patch16_224_msn': 'https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar',
    'vit_large_patch7_224_msn': 'https://dl.fbaipublicfiles.com/msn/vitl7_200ep.pth.tar',
}

NORMALIZATION = {
    'vit_base_patch16_224_mae': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'vit_small_patch16_224_msn': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'vit_large_patch7_224_msn': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
}

MODEL_KWARGS = {
    'vit_base_patch16_224_mae': dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
    ), 
    'vit_small_patch16_224_msn': dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
    ),
    'vit_large_patch7_224_msn': dict(
        patch_size=7, embed_dim=1024, depth=24, num_heads=16,
    )
}

class FeatureModel(nn.Module):

    def __init__(
        self,
        image_size: int = 224,
        model_name: str = 'vit_small_patch16_224_mae',
        global_pool: str = '',  # '' or 'token'
    ) -> None:
        
        super().__init__()
        self.model_name = model_name

        if self.model_name == 'identity':
            return
        
        # Create model
        self.model = VisionTransformer(
            img_size=image_size, num_classes=0, global_pool=global_pool,
            **MODEL_KWARGS[model_name])

        # Model properties
        self.feature_dim = self.model.embed_dim
        self.mean, self.std = NORMALIZATION[model_name]

        # Load pretrained checkpoint
        checkpoint = torch.hub.load_state_dict_from_url(MODEL_URLS[model_name])
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # NOTE: Comment the line below if using the projection head, uncomment if not using it
            # See https://github.com/facebookresearch/msn/blob/81cb855006f41cd993fbaad4b6a6efbb486488e6/src/msn_train.py#L490-L502
            # for more info about the projection head
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        else:
            raise NotImplementedError()
        state_dict['pos_embed'] = resize_pos_embed(state_dict['pos_embed'], self.model.pos_embed)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.fc = nn.Identity()
    
    def denormalize(self, img: Tensor):
        img = TVF.normalize(
        img, 
        mean=[-m/s for m, s in zip(self.mean, self.std)], 
        std=[1/s for s in self.std]
        )
        return torch.clip(img, 0, 1)

    def normalize(self, img: Tensor):
        return TVF.normalize(img, mean=self.mean, std=self.std)
    
    def forward(
        self, 
        x: Tensor, 
        return_type: str = 'features',
        return_upscaled_features: bool = True,
        return_projection_head_output: bool = False,
    ):
        """Normalizes the input `x` and runs it through `model` to obtain features"""
        assert return_type in {'cls_token', 'features', 'all'}

        # Identity
        if self.model_name == 'identity':
            return x
        
        # Normalize and forward
        B, C, H, W = x.shape
        x = self.normalize(x)
        feats = self.model(x)

        # Reshape to image-like size
        if return_type in {'features', 'all'}:
            B, T, D = feats.shape
            assert math.sqrt(T - 1).is_integer()
            HW_down = int(math.sqrt(T - 1))  # subtract one for CLS token
            output_feats: Tensor = feats[:, 1:, :].reshape(B, HW_down, HW_down, D).permute(0, 3, 1, 2)  # (B, D, H_down, W_down)
            if return_upscaled_features:
                output_feats = F.interpolate(output_feats, size=(H, W), mode='bilinear',
                    align_corners=False)  # (B, D, H_orig, W_orig)

        # Head for MSN
        output_cls = feats[:, 0]
        if return_projection_head_output and return_type in {'cls_token', 'all'}:
            output_cls = self.fc(output_cls)
        
        # Return
        if return_type == 'cls_token':
            return output_cls
        elif return_type == 'features':
            return output_feats
        else:
            return output_cls, output_feats


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def save_feature_maps(features: torch.Tensor, save_dir: str, max_channels: int = 5):
        os.makedirs(save_dir, exist_ok=True)
        features = features.squeeze(0).cpu()  # (D, H, W)

        for i in range(min(max_channels, features.shape[0])):
            feat_map = features[i]
            feat_map -= feat_map.min()
            feat_map /= feat_map.max() + 1e-5
            plt.imsave(os.path.join(save_dir, f'feature_{i}.png'), feat_map.numpy(), cmap='viridis')
            print(f"Saved: {save_dir}/feature_{i}.png")
    
    image_path = '/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/3D-Future-Demo/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/image.jpg'
    image = PIL.Image.open(image_path).convert("RGB")
    image = TVF.to_tensor(image).unsqueeze(0)  # (1, 3, H, W)

    assert image.size(2) == image.size(3), 'wrong image size'
    image_size = image.size(2)
    device = torch.device("cuda")
    model = FeatureModel(image_size=image_size, model_name="vit_small_patch16_224_msn").to(device)

    with torch.no_grad():
        image = image.to(device)
        features = model(image, return_type='features')  # (1, D, H, W)
        print(features.shape)
    
    save_feature_maps(features, "/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/3D-Future-Demo/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d", max_channels=5)


