# Diffusion Probabilistic Models for 3D Point Cloud Generation
<img src="teaser.png" alt="teaser" width="50%" />

[[Paper](https://arxiv.org/abs/2103.01458)] [[Code](https://github.com/luost26/diffusion-point-cloud)]

The official code repository for our CVPR 2021 paper "Diffusion Probabilistic Models for 3D Point Cloud Generation".

## Installation

**[Step 1]** Setup conda environment

```bash
# Create the environment
conda env create -f env.yml
# Activate the environment
conda activate dpm-pc-gen
```

**[Step 2]** Compile the evaluation module

⚠️ Please compile the module using **`nvcc` 10.0**. Errors might occur if you use other versions (for example 10.1). 

💡 You might specify your `nvcc` path [here](https://github.com/luost26/diffusion-point-cloud/blob/9be449f80b1353e6d39010363d4e139e9e532a2c/evaluation/pytorch_structural_losses/Makefile#L9).

```bash
# Please ensure the conda environment `dpm-pc-gen` is activated.
cd ./evaluation/pytorch_structural_losses
make clean
make
# Return to the project directory
cd ../../
```

## Datasets and Pretrained Models

Datasets and pretrained models are available at: https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ

## Training

```bash
# Train an auto-encoder
python train_ae.py 

# Train a generator
python train_gen.py
```

You may specify the value of arguments. Please find the available arguments in the script. 

Note that `--categories` can take `all` (use all the categories in the dataset), `airplane`, `chair` (use a single category), or `airplane,chair` (use multiple categories, separated by commas).

## Testing

```bash
# Test an auto-encoder
python test_ae.py --ckpt ./pretrained/AE_all.pt --categories all

# Test a generator
python test_gen.py --ckpt ./pretrained/GEN_airplane.pt --categories airplane
```

## Citation

```
@inproceedings{luo2021diffusion,
  author = {Luo, Shitong and Hu, Wei},
  title = {Diffusion Probabilistic Models for 3D Point Cloud Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021}
}
```
