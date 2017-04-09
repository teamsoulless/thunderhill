import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

from keras.models import Model
from config import WIDTH, HEIGHT, DEPTH


def hist(dataset, attr, title=None, xlabel=None, ylabel=None, grid=True):
    plt.hist(dataset[attr].values, 100, normed=0, alpha=0.75)
    plt.title(title)
    if grid is not True:
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)
    plt.show()


def plots(images, row=None, col=None, figsize=(8, 6), labels=[], grid=True):
    if (row is None or col is None):
        row = len(images)
        col = 1

    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(row, col, i+1)
        plt.imshow(img)

        if grid is False:
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

        if len(labels) == len(images):
            plt.xlabel(labels[i])

    plt.tight_layout()
    plt.show()


def plotsFeatures(img, model, depth):
    elu = Model(input=model.layers[0].input, output=model.layers[depth].output)
    elu.compile(optimizer='adam', loss='mse')
    out = elu.predict(np.reshape(img, (1, HEIGHT, WIDTH, DEPTH)))
    feats = [cv2.resize(out[0, :, :, i], (WIDTH, HEIGHT)) for i in range(out.shape[-1])]
    col = 4
    row = out.shape[-1]/col
    plots(feats, row, col, figsize=(16, 12), grid=False)


def plotLearningCurve(fname):
    df = pd.read_csv(fname)
    plt.figure(figsize=(16,12))
    plt.plot(df['epoch'], df['loss'])
    plt.plot(df['epoch'], df['val_loss'])
    plt.legend(['loss', 'val'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    pass