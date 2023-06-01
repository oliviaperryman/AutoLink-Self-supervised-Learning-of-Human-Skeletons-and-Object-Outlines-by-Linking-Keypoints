# Self-supervised, robust 3D keypoint detection

## Recommendation for evaluation

Since the code builds off another repo, I recommend using github's pull request feature to compare the changes I made to the code. Each branch builds off the next and only displays the changes made for that experiment.

I created one pull request per experiment:
- 3D: https://github.com/oliviaperryman/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints/pull/1
- 2.5D: https://github.com/oliviaperryman/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints/pull/2
- Learned rotation axis: https://github.com/oliviaperryman/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints/pull/3

## Instructions

### Installation

Install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). Installing from source was successful.

```
conda create -n cs503 python=3.8
conda activate cs503

conda install pytorch=1.13.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -r requirements.txt

pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

```

Changes to the library code to avoid cuda errors:
- Cast matrix to float https://discuss.pytorch.org/t/svd-error-svd-cuda-gesvdj-not-implemented-for-half/132268
- /pytorch3d/ops/points_alignment.py
    - line 331: XYcov = XYcov.float()
    - line 354: R_test = R_test.float()


### Data
Dowload data: 

https://www.epfl.ch/labs/cvlab/data/data-pose-index-php/

https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

Create images with no background:

```
pip install rembg
rembg p cars/epfl-gims08/tripod-seq cars/no_background_isnet -m isnet-general-use
```

## Train

Run the following command to train the model:

```python train.py --n_parts 8 --missing 0.9 --block 16 --thick "2.5e-3" --sklr 512 --data_root "cars/epfl-gims08/tripod-seq" --dataset "cars_multiview" --batch_size 16```

For no background, use
```data_root "cars/no_background_isnet"```

## File structure

- cars/: folder for storing datasets
- datasets/
    - `cars_multiview.py`: dataset class for epfl multiview cars
    - `cars_multiview_mixed.py`: dataset class for EPFL multiview cars plus Stanford cars
- models/
    - `model.py`: overall model architecture
    - `encoder.py`: encoder architecture including keypoint detector
    - `decoder.py`: decoder architecture
- scratch/: scratch code for testing
- `app.py`: interactive testing app
- `train.py`: main file to run training
- `test.py`: main file for testing

## Experiments structure

Checkout different branches for different experiments.

3D: `git checkout operryman/3d`
2.5D: `git checkout operryman/2.5d`
learned rotation axis: `git checkout operryman/predict_rotation`
