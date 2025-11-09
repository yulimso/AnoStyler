import os
import argparse
import yaml

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import cv2
from PIL import Image
from torchvision import transforms, models, utils as vutils
from torchvision.transforms.functional import adjust_contrast

import clip
from tqdm import tqdm

from src import StyleNet, utils
from src.def_train import *
from src.meta_shape_priors import generate_meta_mask
from src.sam import init_sam_model, combine_with_foreground_mask
from src.two_class_prompt_template import *


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def main(cfg):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If you want to generate different masks for each run,
    # keep these lines commented out.
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    category = cfg["category"]
    defect = cfg["defect"]
    normal_image = cfg["normal_image"]
    num_gen = cfg["num_gen"]
    save_path = cfg["save_path"]
    img_width = cfg["img_width"]
    img_height = cfg["img_height"]
    m_max = cfg["m_max"]
    alpha = cfg["alpha"]
    fg_mask = cfg["fg_mask"]
    sam_ckpt = cfg["sam_ckpt"]
    crop_size = cfg["crop_size"]
    num_crops = cfg["num_crops"]
    max_step = cfg["max_step"]
    lr = cfg["lr"]
    thresh = cfg["thresh"]
    lambda_tv = cfg["lambda_tv"]
    lambda_pdir = cfg["lambda_pdir"]
    lambda_gdir = cfg["lambda_gdir"]
    lambda_c = cfg["lambda_c"]
    lambda_mclip = cfg["lambda_mclip"]


    predictor = None
    if fg_mask:
        predictor = init_sam_model(
            model_type="vit_b",
            checkpoint_path=sam_ckpt
        )

    VGG = models.vgg19(pretrained=True).features.to(device).eval()
    for p in VGG.parameters():
        p.requires_grad = False
    clip_model, _ = clip.load('ViT-B/32', device)

    category_path = os.path.join(save_path, category, defect)
    os.makedirs(category_path, exist_ok=True)

    image_dir = os.path.join(category_path, "image")
    os.makedirs(image_dir, exist_ok=True)

    normal_phrases = [template_prompt.format(normal_prompt.format(category))
                      for template_prompt in template_level_prompts
                      for normal_prompt in state_level_normal_prompts]

    abnormal_phrases = []
    for template_prompt in template_level_prompts:
        for abnormal_prompt in state_level_abnormal_prompts + state_level_abnormal_prompts:
            try:
                phrase = template_prompt.format(abnormal_prompt.format(category, defect))
            except TypeError:
                phrase = template_prompt.format(abnormal_prompt.format(category))
            abnormal_phrases.append(phrase)

    with torch.no_grad():
        tokens = clip.tokenize(abnormal_phrases).to(device)
        text_features = clip_model.encode_text(tokens).mean(dim=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        tokens_src = clip.tokenize(normal_phrases).to(device)
        text_source = clip_model.encode_text(tokens_src).mean(dim=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)

    content_image = utils.load_image2(
        normal_image, img_height=img_height, img_width=img_width
    ).to(device)

    source_features = clip_model.encode_image(clip_normalize(content_image)).detach()
    source_features /= source_features.norm(dim=-1, keepdim=True)

    for i in tqdm(range(num_gen), desc=f"Generating {num_gen} images"):

        if fg_mask:
            mask = combine_with_foreground_mask(
                image_path=normal_image,
                predictor=predictor,
                m_max=m_max,
                alpha=alpha
            )
        else:
            mask = generate_meta_mask(
                W=img_width,
                H=img_height,
                m_max=m_max,
                alpha=alpha
            )
            while mask.sum() == 0:
                mask = generate_meta_mask(
                    W=img_width,
                    H=img_height,
                    m_max=m_max,
                    alpha=alpha
                )

        mask_to_save = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_dir = os.path.join(category_path, "mask")
        os.makedirs(mask_dir, exist_ok=True)
        temp_mask_path = os.path.join(mask_dir, f"gen_mask_{i}.jpg")
        cv2.imwrite(temp_mask_path, mask_to_save)

        save_img_path = os.path.join(image_dir, f"gen_ano_{i}.jpg")

        run_style_transfer(
            temp_mask_path, content_image,
            clip_model=clip_model, VGG=VGG, device=device,
            img_height=img_height, img_width=img_width,
            lambda_tv=lambda_tv, lambda_pdir=lambda_pdir,
            lambda_gdir=lambda_gdir, lambda_c=lambda_c,
            lambda_mclip=lambda_mclip,
            crop_size=crop_size, num_crops=num_crops,
            max_step=max_step, lr=lr, thresh=thresh,
            save_img_path=save_img_path,
            source_features=source_features,
            text_features=text_features,
            text_source=text_source
        )


if __name__ == "__main__":
    cfg = load_config()
    main(cfg)