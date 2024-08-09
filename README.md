# GaussTR

## Setup

### Installation

```
pip install -r requirements

pip install -v -e submodules/diff-gaussian-rasterization
pip install -v -e .
```

### Datasets

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

Download `gts` from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction),
the folder is arranged as:

```shell script
└── OccVision
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── gts (new)
            ├── nuscenes_infos_train.pkl (new)
            └── nuscenes_infos_val.pkl (new)
```

## Usage

### Training

```
PYTHONPATH=. mim train mmdet3d configs/gausstr/gausstr.py -l pytorch -G 8
```

### Testing

```
PYTHONPATH=. mim test mmdet3d configs/gausstr/gausstr.py -C [checkpoint]
```

### Visualization

Run the testing with `DumpVisualizationHook`, then

```
python tools/visualize_occ.py [pkl_path]
```
