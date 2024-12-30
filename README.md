# [GaussTR](): Foundation Model-Aligned [Gauss]()ian [Tr]()ansformer for Self-Supervised 3D Spatial Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2412.13193-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2412.13193)
[![License: MIT](https://img.shields.io/github/license/hustvl/GaussTR)](LICENSE)

## Setup

### Installation

```
pip install -r requirements.txt
```

### Dataset Preparation

1. Prepare the nuScenes dataset following the instructions from [mmdetection3d](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html#nuscenes).
2. Update the dataset with `scene_idx` to match with the occupancy ground truths by running:

    ```
    python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
    ```

3. Download occupancy ground truth data from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and place them under `data/nuscenes/gts`.
4. Generate features and rendering targets using [Metric 3D V2](https://github.com/YvanYin/Metric3D), [FeatUp](https://github.com/mhamilton723/FeatUp) for MaskCLIP, and [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2).

### CLIP Text Embeddings

Directly download from the releases, or manually generate CLIP text embeddings for custom categories by referring to https://github.com/open-mmlab/mmpretrain/pull/1737.

## Usage

### Training

```
PYTHONPATH=. mim train mmdet3d configs/gausstr/gausstr.py [-l pytorch -G [GPU_NUM]]
```

### Testing

```
PYTHONPATH=. mim test mmdet3d configs/gausstr/gausstr.py -C [CKPT_PATH] [-l pytorch -G [GPU_NUM]]
```

### Visualization

After testing with `DumpResultHook`, visualize the results by executing:

```
python tools/visualize.py [PKL_PATH] [--save]
```

## Citation

If you find our paper and code helpful for your research, please consider starring this repository :star: and citing our work:

```BibTeX
@article{GaussTR,
    title = {GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding}, 
    author = {Haoyi Jiang and Liu Liu and Tianheng Cheng and Xinjie Wang and Tianwei Lin and Zhizhong Su and Wenyu Liu and Xinggang Wang},
    year = 2024,
    journal = {arXiv preprint arXiv:2412.13193}
}
```

## Acknowledgements

This project builds upon the pioneering work of [FeatUp](https://github.com/mhamilton723/FeatUp), [MaskCLIP](https://github.com/chongzhou96/MaskCLIP) and [gsplat](https://github.com/nerfstudio-project/gsplat).  We extend our gratitude to these projects for their contributions to the community.

## License

Released under the [MIT](LICENSE) License.
