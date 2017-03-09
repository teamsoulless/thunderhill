import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import Model


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

    out = elu.predict(np.reshape(img, (1, 66, 200, 3)))
    feats = [cv2.resize(out[0, :, :, i], (200, 66)) for i in range(out.shape[-1])]
    col = 4
    row = out.shape[-1]/col
    plots(feats, row, col, figsize=(16, 12), grid=False)


if __name__ == '__main__':
    pass