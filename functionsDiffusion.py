import numpy as np
import scipy
from scipy.ndimage import laplace
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm, animation, rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import time
import os
from matplotlib.backends.backend_pdf import PdfPages


def ficks(s, w):
    return(laplace(s) / np.power(w, 2))


def hill(s, K, lam):
    h = s**lam / (K**lam + s**lam)
    return(h)


#def nutrient(N_matrix):
#    return (hill(N_matrix, 80, 2))


def multi_plots(sim, title=""):
    f, ax = plt.subplots(3, 3, sharex=True, sharey=False, figsize=(15, 15))

    f.suptitle(title, fontsize=40)
    im1 = ax[0, 0].imshow(sim[3], interpolation="none", cmap=cm.viridis, vmin=0, vmax=1)
    ax[0, 0].set_title("Sender")
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im1, cax=cax, shrink=0.8)

    im2 = ax[0, 1].imshow(sim[5], interpolation="none", cmap=cm.viridis, vmin=0, vmax=1)
    ax[0, 1].set_title("Receiver")
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im2, cax=cax, shrink=0.8)

    im3 = ax[0, 2].imshow(sim[1], interpolation="none", cmap=cm.viridis, vmin=0, vmax=5)
    ax[0, 2].set_title("Arabinose")
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im3, cax=cax, shrink=0.8)

    im4 = ax[1, 0].imshow(sim[3], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1, 0].set_title("LuxI")
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im4, cax=cax, shrink=0.8)

    im5 = ax[1, 1].imshow(sim[4], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1, 1].set_title("C6")
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im5, cax=cax, shrink=0.8)

    im6 = ax[1, 2].imshow(sim[6], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1, 2].set_title("GFP")
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im6, cax=cax, shrink=0.8)

    im7 = ax[2, 0].imshow(sim[2], interpolation="none", cmap=cm.viridis, vmin=0, vmax=100)
    ax[2, 0].set_title("Nutrients")
    divider = make_axes_locatable(ax[2, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im7, cax=cax, shrink=0.8)

    ax[2, 1].axis('off')
    ax[2, 2].axis('off')

    return(f)


def plots(sim, names):

    n_plots = sim.shape[0]
    x = int(np.ceil(n_plots / 3))

    f, ax = plt.subplots(x, 3, sharex=True, sharey=False, figsize=(15, 15))

    for i, val in enumerate(ax.flatten()):

        if i < n_plots:
            im = ax.flatten()[i].imshow(sim[i], cmap=cm.viridis, vmin=0)
            ax.flatten()[i].set_title(names[i])
            divider = make_axes_locatable(ax.flatten()[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            f.colorbar(im, cax=cax, shrink=0.8, label='')
        else:
            ax.flatten()[i].axis("off")
