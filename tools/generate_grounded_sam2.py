# Refered to https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_local_demo.py
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_convert
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_image, load_model, predict

OCC3D_CATEGORIES = (
    ['barrier', 'concrete barrier', 'metal barrier', 'water barrier'],
    ['bicycle', 'bicyclist'],
    ['bus'],
    ['car'],
    ['crane'],
    ['motorcycle', 'motorcyclist'],
    ['pedestrian', 'adult', 'child'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['traffic island', 'rail track', 'lake', 'river'],
    ['sidewalk'],
    ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
    ['building', 'wall', 'guard rail', 'fence', 'pole', 'drainage',
     'hydrant', 'street sign', 'traffic light'],
    ['tree', 'bush'],
    ['sky', 'empty'],
)
CLASSES = sum(OCC3D_CATEGORIES, [])
TEXT_PROMPT = '. '.join(CLASSES)
INDEX_MAPPING = [
    outer_index for outer_index, inner_list in enumerate(OCC3D_CATEGORIES)
    for _ in inner_list
]

IMG_PATH = 'data/nuscenes/samples/'
OUTPUT_DIR = Path('nuscenes_grounded_sam2/')

SAM2_CHECKPOINT = 'checkpoints/sam2.1_hiera_base_plus.pt'
SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
GROUNDING_DINO_CONFIG = 'grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py'
GROUNDING_DINO_CHECKPOINT = 'gdino_checkpoints/groundingdino_swinb_cogcoor.pth'

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE)

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT

for view_dir in os.listdir(IMG_PATH):
    for image_path in tqdm(os.listdir(osp.join(IMG_PATH, view_dir))):
        image_source, image = load_image(
            os.path.join(IMG_PATH, view_dir, image_path))

        sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(
            boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if input_boxes.shape[0] != 0:
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

        results = np.zeros_like(masks[0])
        if input_boxes.shape[0] != 0:
            for i in range(len(labels)):
                if labels[i] not in CLASSES:
                    continue
                pred = INDEX_MAPPING[CLASSES.index(labels[i])] + 1
                results[masks[i].astype(bool)] = pred

        np.save(osp.join(OUTPUT_DIR, image_path.split('.')[0]), results)
