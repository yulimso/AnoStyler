import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from .meta_shape_priors import generate_meta_mask
import os


def init_sam_model(model_type='vit_b', checkpoint_path='sam_vit_b_01ec64.pth'):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    return SamPredictor(sam)


def get_background_only_mask(image, predictor):
    h, w, _ = image.shape

    bg_points = np.array([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])
    bg_labels = np.array([0, 0, 0, 0]) 

    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=bg_points,
        point_labels=bg_labels,
        multimask_output=True
    )

    best_mask = masks[0]
    inverted_mask = 1 - best_mask        
    inverted_mask = (inverted_mask * 255).astype(np.uint8) 

    return inverted_mask


def combine_with_foreground_mask(image_path, predictor, m_max=5, alpha=0.7, out_size=(512, 512)):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_W, base_H = 256, 256  

    while True:
        mask_256 = generate_meta_mask(W=base_W, H=base_H, m_max=m_max, alpha=alpha)
        fg_mask = get_background_only_mask(image, predictor)
        fg_mask_256 = cv2.resize(fg_mask, (base_W, base_H), interpolation=cv2.INTER_NEAREST)

        if fg_mask_256.dtype != np.uint8:
            fg_mask_256 = fg_mask_256.astype(np.uint8)

        combined_mask_256 = cv2.bitwise_and(mask_256, fg_mask_256)

        if np.any(combined_mask_256 > 0):
            final_mask = cv2.resize(combined_mask_256, out_size, interpolation=cv2.INTER_NEAREST)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

            return final_mask