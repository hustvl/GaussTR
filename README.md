<div align="center">

# [GaussTR](): Foundation Model-Aligned [Gauss]()ian [Tr]()ansformer for Self-Supervised 3D Spatial Understanding

[Haoyi Jiang](https://scholar.google.com/citations?user=_45BVtQAAAAJ)<sup>1</sup>, Liu Liu<sup>2</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ)<sup>1</sup>, Xinjie Wang<sup>2</sup>,
[Tianwei Lin](https://wzmsltw.github.io/)<sup>2</sup>, Zhizhong Su<sup>2</sup>, Wenyu Liu<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1</sup><br>
<sup>1</sup>Huazhong University of Science & Technology, <sup>2</sup>Horizon Robotics

[![Project page](https://img.shields.io/badge/project%20page-hustvl.github.io%2FGaussTR-blue)](https://hustvl.github.io/GaussTR/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.13193-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2412.13193)
[![License: MIT](https://img.shields.io/github/license/hustvl/GaussTR)](LICENSE)

</div>

## Setup

### Installation

We recommend cloning the repository with the `--single-branch` option to avoid downloading unnecessary large media files for the project website from other branches.

```bash
git clone https://github.com/hustvl/GaussTR.git --single-branch
cd GaussTR
pip install -r requirements.txt
```

### Dataset Preparation

1. Prepare the nuScenes dataset following the instructions provided in the [mmdetection3d docs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html#nuscenes).
2. Update the dataset `.pkl` files with `scene_idx` to match with occupancy ground truths by running:

    ```bash
    python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
    ```

3. Download the occupancy ground truth data from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and place it under `data/nuscenes/gts`.
4. Generate the required features and rendering targets:

    * Run `PYTHONPATH=. python tools/generate_depth.py` to generate metric depth estimations.
    * Navigate to the [FeatUp](https://github.com/mhamilton723/FeatUp) repository and run `python tools/generate_featup.py` there.
    * Optionally, navigate to the [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) and run `python tools/generate_grounded_sam2.py` to enable training augmentation.

### CLIP Text Embeddings

Download pre-generated CLIP text embeddings from the [releases section](https://github.com/hustvl/GaussTR/releases/), or manually generate custom embeddings by referring to https://github.com/open-mmlab/mmpretrain/pull/1737.

## Usage

### Training

```bash
PYTHONPATH=. mim train mmdet3d configs/gausstr.py [-l pytorch -G [GPU_NUM]]
```

### Testing

```bash
PYTHONPATH=. mim test mmdet3d configs/gausstr.py -C [CKPT_PATH] [-l pytorch -G [GPU_NUM]]
```

### Visualization

To enable visualization during testing, include the following in the config:

```python
custom_hooks = [
    dict(type='DumpResultHook '),
]
```

After testing, visualize the saved `.pkl` files by executing:

```bash
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
