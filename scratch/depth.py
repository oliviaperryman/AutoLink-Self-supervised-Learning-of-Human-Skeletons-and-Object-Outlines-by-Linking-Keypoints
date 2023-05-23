import os

import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from tqdm import tqdm

checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint)


if __name__ == "__main__":
    root = "../cars/"
    tripod = "epfl-gims08/tripod-seq"
    depth = "depth"

    all_cars = sorted(os.listdir(os.path.join(root, tripod)))

    for car in tqdm(all_cars):
        img = Image.open(os.path.join(root, tripod, car))
        predictions = depth_estimator(img)
        predictions["depth"].save(os.path.join(root, depth, car))
