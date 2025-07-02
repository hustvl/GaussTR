<div align="center">

# [GaussTR](): Foundation Model-Aligned [Gauss]()ian [Tr]()ansformer for Self-Supervised 3D Spatial Understanding

[Haoyi Jiang](https://scholar.google.com/citations?user=_45BVtQAAAAJ)<sup>1</sup>, Liu Liu<sup>2</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ)<sup>1</sup>, Xinjie Wang<sup>2</sup>,
[Tianwei Lin](https://wzmsltw.github.io/)<sup>2</sup>, Zhizhong Su<sup>2</sup>, Wenyu Liu<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1</sup><br>
<sup>1</sup>Huazhong University of Science & Technology, <sup>2</sup>Horizon Robotics

[**CVPR 2025**]()

[![Project page](https://img.shields.io/badge/project%20page-hustvl.github.io%2FGaussTR-blue)](https://hustvl.github.io/GaussTR/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.13193-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2412.13193)
[![License: MIT](https://img.shields.io/github/license/hustvl/GaussTR)](LICENSE)

</div>

## News

* ***Feb 27 '25:*** Our paper has been accepted at CVPR 2025. 🎉
* ***Feb 11 '25:*** Released the model integrated with Talk2DINO, achieving new state-of-the-art results.
* ***Dec 17 '24:*** Released our arXiv paper along with the source code.

## Setup

### Installation

We recommend cloning the repository using the `--single-branch` option to avoid downloading unnecessary large media files for the project website from other branches:

```bash
git clone https://github.com/hustvl/GaussTR.git --single-branch
cd GaussTR
pip install -r requirements.txt
```

### Dataset Preparation

1. Download or manually prepare the nuScenes dataset following the instructions in the [mmdetection3d docs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html#nuscenes) and place it in `data/nuscenes`.
   **NOTE:** Please be aware that we are using the latest OpenMMLab V2.0 format. If you've previously prepared the nuScenes dataset from other repositories, it might be outdated. For more information, please refer to [update_infos_to_v2.py](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/update_infos_to_v2.py).
2. **Update the prepared dataset `.pkl` files with the `scene_idx` field to match the occupancy ground truths:**

    ```bash
    python tools/update_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
    ```

3. Download the occupancy ground truth data from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and place it in `data/nuscenes/gts`.
4. Generate features and rendering targets:

    * Run `PYTHONPATH=. python tools/generate_depth.py` to generate metric depth estimations.
    * **[For GaussTR-FeatUp Only]** Navigate to the [FeatUp](https://github.com/mhamilton723/FeatUp) repository and run `python tools/generate_featup.py`.
    * **[Optional for GaussTR-FeatUp]** Navigate to the [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) and run `python tools/generate_grounded_sam2.py` to enable auxiliary segmentation supervision.

### CLIP Text Embeddings

Download the pre-generated CLIP text embeddings from the [Releases](https://github.com/hustvl/GaussTR/releases/) page.  Alternatively, you can generate custom embeddings by referring to [mmpretrain #1737](https://github.com/open-mmlab/mmpretrain/pull/1737) or [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO).

**Tip:** The default prompts have not been delicately tuned. Customizing them may yield improved results.

## Usage

|                               Model                               |  IoU  |  mIoU |                                                 Checkpoint                                                 |
| ----------------------------------------------------------------- | ----- | ----- | ---------------------------------------------------------------------------------------------------------- |
| [GaussTR-FeatUp](configs/gausstr_featup.py)                       | 45.19 | 11.70 | [checkpoint](https://github.com/hustvl/GaussTR/releases/download/v1.0/gausstr_featup_e24_miou11.70.pth)    |
| [GaussTR-Talk2DINO](configs/gausstr_talk2dino.py)<sup>*New*</sup> | 44.54 | 12.27 | [checkpoint](https://github.com/hustvl/GaussTR/releases/download/v1.0/gausstr_talk2dino_e20_miou12.27.pth) |

### Training

**Tip:** Due to the current lack of optimization for voxelization operations, evaluation during training can be time-consuming. To accelerate training, consider evaluating using the `mini_train` set or reducing the evaluation frequency.

```bash
PYTHONPATH=. mim train mmdet3d [CONFIG] [-l pytorch -G [GPU_NUM]]
```

### Testing

```bash
PYTHONPATH=. mim test mmdet3d [CONFIG] -C [CKPT_PATH] [-l pytorch -G [GPU_NUM]]
```

### Visualization

To enable visualization, run the testing with the following included in the config:

```python
custom_hooks = [
    dict(type='DumpResultHook'),
]
```

After testing, visualize the saved `.pkl` files with:

```bash
python tools/visualize.py [PKL_PATH] [--save]
```

## Citation

If our paper and code contribute to your research, please consider starring this repository :star: and citing our work:

```BibTeX
@inproceedings{GaussTR,
    title     = {GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding},
    author    = {Haoyi Jiang and Liu Liu and Tianheng Cheng and Xinjie Wang and Tianwei Lin and Zhizhong Su and Wenyu Liu and Xinggang Wang},
    year      = 2025,
    booktitle = {CVPR}
}
```

## Acknowledgements

This project is built upon the pioneering work of [FeatUp](https://github.com/mhamilton723/FeatUp), [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO), [MaskCLIP](https://github.com/chongzhou96/MaskCLIP) and [gsplat](https://github.com/nerfstudio-project/gsplat). We extend our gratitude to these projects for their contributions to the community.

## License

Released under the [MIT](LICENSE) License.
