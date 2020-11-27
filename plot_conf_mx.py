import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_confusion_matrix(y_true, y_pred,
                          labels,
                          normalize, title, save, precise,
                          cmap=plt.cm.Blues
                          ):
    # Removes labels not in trgs or outs.
    labels = [label for label in labels if label in (y_true + y_pred)]

    # Computes the conf mx.
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # --- plot 크기 조절 --- #
    plt.rcParams['savefig.dpi'] = 60  # dpi: Dots per inch.
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['figure.figsize'] = [20, 20]  # plot 크기
    plt.rcParams.update({'font.size': 13})
    # --- plot 크기 조절 --- #

    fig, ax = plt.subplots()
    # fig.tight_layout()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # --- bar 크기 조절 --- #
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # --- bar 크기 조절 --- #
    # ax.figure.colorbar(im, ax=ax)

    # Plots ticks.
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='Targets', xlabel='Outputs')
    # Rotates the tick labels and sets their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loops over data dimensions and creates text annotations.
    if not precise:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    if save:
        plt.savefig(f"{title}.png" if title else f"conf_mx{time.strftime('%mm%dd%HH%MM')}.png")

    plt.show()

