import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models, utils as vutils
import clip
from . import StyleNet
from . import utils
import argparse
from torchvision.transforms.functional import adjust_contrast
from tqdm import tqdm


def print_parameter_count(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] {name} — Total: {total_params:,}, Trainable: {trainable_params:,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numeric_key(fname):
    return int(os.path.splitext(fname)[0])

def img_normalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
    return (image - mean) / std

def clip_normalize(image):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(1, -1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(1, -1, 1, 1)
    return (image - mean) / std

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    return torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)


def run_style_transfer(mask_path, content_image, 
                        clip_model, VGG, device,
                        img_height, img_width,
                        lambda_tv, lambda_pdir, lambda_gdir, lambda_c,lambda_mclip,
                        crop_size, num_crops, max_step, lr, thresh,
                        save_img_path, source_features, text_features, text_source):

    mask = np.array(Image.open(mask_path).convert("L")) / 255.0
    if np.max(mask) == 0:
        content_image_resized = F.interpolate(content_image, size=(256, 256), mode='bilinear', align_corners=False)
        vutils.save_image(content_image_resized, save_img_path, normalize=False)
        return

    mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    mask = F.interpolate(mask, size=(img_height, img_width), mode='nearest')

    style_net = StyleNet.UNet().to(device)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    optimizer = optim.Adam(style_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    content_features = utils.get_features(img_normalize(content_image), VGG)

    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])

    for epoch in tqdm(range(max_step + 1)):
        full_output = style_net(content_image, use_sigmoid=True)
        target = content_image * (1 - mask) + full_output * mask
        target.requires_grad_(True)

        target_features = utils.get_features(img_normalize(target), VGG)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        _, _, H, W = target.shape
        img_proc_anom, img_proc_norm = [], []
        crop_coords = [] 

        for _ in range(num_crops):
            top = torch.randint(0, H - crop_size + 1, (1,))
            left = torch.randint(0, W - crop_size + 1, (1,))
            crop_coords.append((top.item(), left.item()))

            patch_anom = target[:, :, top:top+crop_size, left:left+crop_size]
            patch_anom = F.interpolate(patch_anom, size=224, mode='bilinear', align_corners=False)
            patch_anom = augment(patch_anom)
            img_proc_anom.append(patch_anom)

        img_anom = torch.cat(img_proc_anom, dim=0) 

        feat_anom = clip_model.encode_image(clip_normalize(img_anom))
        feat_anom = feat_anom / feat_anom.norm(dim=-1, keepdim=True)

        delta_I = feat_anom - source_features
        delta_I = delta_I / delta_I.norm(dim=-1, keepdim=True)

        delta_T = (text_features - text_source).repeat(num_crops, 1)
        delta_T = delta_T / delta_T.norm(dim=-1, keepdim=True)

        loss_temp = 1 - torch.cosine_similarity(delta_I, delta_T, dim=1)
        loss_temp[loss_temp < thresh] = 0

        weights = []
        for top, left in crop_coords:
            mask_crop = mask[:, top:top+crop_size, left:left+crop_size]
            patch_ratio = mask_crop.float().mean().item()
            weights.append(patch_ratio)

        weights = torch.tensor(weights).to(device)
        if weights.sum() > 1e-8:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)

        loss_patch = (weights * loss_temp).sum()


        glob_features = clip_model.encode_image(clip_normalize(target))
        glob_features = glob_features / glob_features.norm(dim=-1, keepdim=True)
        glob_direction = glob_features - source_features
        glob_direction = glob_direction / glob_direction.norm(dim=-1, keepdim=True)
        gtext_direction = (text_features - text_source)
        gtext_direction = gtext_direction / gtext_direction.norm(dim=-1, keepdim=True)
        loss_glob = (1 - torch.cosine_similarity(glob_direction, gtext_direction)).mean()

        reg_tv = lambda_tv * get_image_prior_losses(target)
        
        masked_target = target * mask 
        masked_target_resized = F.interpolate(masked_target, size=224, mode='bicubic')
        masked_clip_feat = clip_model.encode_image(clip_normalize(masked_target_resized))
        masked_clip_feat = masked_clip_feat / masked_clip_feat.norm(dim=-1, keepdim=True)
        loss_clip_sim = 1 - torch.cosine_similarity(masked_clip_feat, text_features).mean()
        
        total_loss = lambda_pdir * loss_patch + lambda_c * content_loss + reg_tv + lambda_gdir * loss_glob + lambda_mclip*loss_clip_sim

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    output = torch.clamp(target.clone(), 0, 1)
    output_resized = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
   
    vutils.save_image(output_resized, save_img_path, normalize=False)
