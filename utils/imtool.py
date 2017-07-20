import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np


def gan_grid_plot(samples, show=False):
  # Check samples' shape to make sure they can be shown in plt
  if samples.shape[-1] == 1:
    samples = samples.reshape(samples.shape[:-1])

  # Plot samples
  manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
  manifold_w = int(np.floor(np.sqrt(samples.shape[0])))

  fig = plt.figure(figsize=(manifold_h, manifold_w))
  gs = gridspec.GridSpec(manifold_h, manifold_w)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample, cmap='Greys_r')

  plt.tight_layout()
  fig.subplots_adjust(wspace=0.03, hspace=0.03)

  if show:
    plt.show()

  return fig
