# Diffusion-Point-Cloud (PointCNN version)

This project is based on the open source implementation of the paper [**“Diffusion Probabilistic Models for 3D Point Cloud Generation”**](https://arxiv.org/abs/2103.01458), extending its original version and replacing the **backbone** of point cloud feature extraction from **PointNet** to **PointCNN**. This version achieves better generation quality and diversity on several 3D point cloud datasets.

---

## Project Introduction

In the original paper, the authors applied diffusion probabilistic models to the task of 3D point cloud generation and proposed a denoising model based on PointNet as a feature extraction network. By defining the forward denoising process in the training phase, the model learns the inverse denoising process, so that high-fidelity target point clouds can be gradually sampled from Gaussian noise point clouds during inference.

However, although **PointNet** is simple and effective to implement, its ability to express local structures is relatively limited. To this end, we replaced **PointNet** with [**PointCNN**](https://arxiv.org/abs/1801.07791) to enhance the ability to extract local neighborhood geometric information, thereby achieving better performance in generating finer local details and shape diversity.

---

## Major updates

1. **Feature extraction network: switch from PointNet to PointCNN**
- **PointCNN** introduces the X-Conv operation, first performs a learnable transformation on the neighborhood point set, and then performs a convolution-like aggregation, so that the model can better capture the local geometric structure and the relationship between points.
- Compared with PointNet, which only uses MLP for each point and performs global pooling, PointCNN can more effectively retain and integrate local-global information and improve the representation of complex 3D shapes.
- With this change, the model has better performance in local detail restoration and generation diversity.

2. **Training stability**
- Further optimize the hyperparameters, including batch size, learning rate, etc., to adapt to the deep network structure of PointCNN.
- Experiments show that there is a certain degree of improvement in common evaluation indicators (such as Coverage, MMD, Chamfer Distance, etc.).

3. **Overall performance improvement**
- Compared with the original PointNet version, the generated 3D point cloud is more realistic and natural in both overall structure and local details.

---

## Environment requirements

- Python 3.7+
- PyTorch >= 1.7 (compatible with CUDA 10.2 / 11.x)
- Common scientific computing and visualization libraries such as Numpy, Scipy, Matplotlib
- [Open3D](http://www.open3d.org/) (optional, used for point cloud operations, etc.)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (if your PointCNN implementation relies on PyG's neighborhood search and other functions)

**[Option 1]** Please first install the required libraries according to [env.yml](./env.yml) in this repository or according to the dependencies listed in the main branch:
```bash
# Create the environment
conda env create -f env.yml
# Activate the environment
conda activate dpm-pc-gen
```
**[Option 2]** Or you may setup the environment manually (**If you are using GPUs that only work with CUDA 11 or greater**).

Our model only depends on the following commonly used packages, all of which can be installed via conda.

| Package      | Version                          |
| ------------ | -------------------------------- |
| PyTorch      | ≥ 1.7.0                          |
| h5py         | *not specified* (we used 4.61.1) |
| tqdm         | *not specified*                  |
| tensorboard  | *not specified* (we used 2.5.0)  |
| numpy        | *not specified* (we used 1.20.2) |
| scipy        | *not specified* (we used 1.6.2)  |
| scikit-learn | *not specified* (we used 0.24.2) |


## Data preparation

### Dataset

- **It is recommended to use ShapeNet, ModelNet and other common 3D shape datasets for experiments. **
- **Download and unzip the corresponding dataset to the `data/` directory (you can also specify the path yourself) according to actual needs. **

### Preprocessing

- For each 3D object, downsample/normalize it to a fixed number of points (such as 1024 points) as needed, and convert it to `.xyz` or `.npy` format.
- The above steps can be completed in the script `data_preprocess.py`, and the preprocessing results are stored in the specified folder.
- For details, please refer to the main branch and expect to remain consistent

## Configuration file

- Set model hyperparameters, training hyperparameters, dataset path and other information in `configs/pointcnn_config.yaml`.
- Core parameters include:
- `num_points`: The number of points in each point cloud (such as 1024).
- `batch_size`: Training batch size.
- `learning_rate`: Initial learning rate.
- `diffusion_steps`: The number of steps in the diffusion process.
- `model`: Specify **PointCNN** as the feature extraction network.


## About the EMD Metric

We have removed the EMD module due to GPU compatability issues. The legacy code can be found on the `emd-cd` branch.

If you have to compute the EMD score or compare our model with others, we strongly advise you to use your own code to compute the metrics. The generation and decoding results will be saved to the `results` folder after each test run.

## Training

```bash
# Train an auto-encoder
python train_ae.py 

# Train a generator
python train_gen.py
```

You may specify the value of arguments. Please find the available arguments in the script. 

Note that `--categories` can take `all` (use all the categories in the dataset), `airplane`, `chair` (use a single category), or `airplane,chair` (use multiple categories, separated by commas).

### Notes on the Metrics

Note that the metrics computed during the validation stage in the training script (`train_gen.py`, `train_ae.py`) are not comparable to the metrics reported by the test scripts (`test_gen.py`, `test_ae.py`). ***If you train your own models, please evaluate them using the test scripts***. The differences include:
1. The scale of Chamfer distance in the training script is different. In the test script, we renormalize the bounding boxes of all the point clouds before calculating the metrics (Line 100, `test_gen.py`). However, in the validation stage of training, we do not renormalize the point clouds.
2. During the validation stage of training, we only use a subset of the validation set (400 point clouds) to compute the metrics and generates only 400 point clouds (controlled by the `--test_size` parameter). Limiting the number to 400 is for saving time. However, the actual size of the `airplane` validation set is 607, larger than 400. Less point clouds mean that it is less likely to find similar point clouds in the validation set for a generated point cloud. Hence, it would lead to a worse Minimum-Matching-Distance (MMD) score even if we renormalize the shapes during the validation stage in the training script.


## Testing

```bash
# Test an auto-encoder
python test_ae.py --ckpt ./pretrained/AE_all.pt --categories all

# Test a generator
python test_gen.py --ckpt ./pretrained/GEN_airplane.pt --categories airplane
```




## Experimental results and performance

Compared with the original **PointNet** version, **PointCNN** as the backbone network can capture richer local geometric structures, thus achieving improvements in **Coverage (COV)**, **Minimum Matching Distance (MMD)**, and **1-NNA** indicators:

The following are the local test results of the current setting on the Airplane data:

| Method                  | COV-CD (↑) | COV-EMD (↑) | MMD-CD (↓) | MMD-EMD (↓) | 1-NNA-CD (↓) | 1-NNA-EMD (↓) |
|-----------------------|-----------|------------|-----------|------------|--------------|---------------|
| **PointNet (Original)**   | 48.71%     | 45.47%          | 3.276    | 1.061          | 64.83%     | 75.12%             |
| **PointCNN (This project)** |  48.83%   |  45.60%        |  3.109   |  0.998     |   64.56%    |  75.05%       |


## References

- [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://arxiv.org/abs/2103.01458)
Shitong Luo, Wei Hu

- [PointCNN: Convolution On X-Transformed Points](https://arxiv.org/abs/1801.07791)
Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, Baoquan Chen

## Acknowledgements

- Thanks to the original open source project author for providing the basic framework and reference implementation.
- Thanks to all developers who have contributed to the open source community.

If you encounter any problems while using or reproducing this project, please [submit an issue](https://github.com/luost26/diffusion-point-cloud/issues) or contact the author.

