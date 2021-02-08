try:
  import matplotlib.gridspec as gridspec
  import matplotlib.pyplot as plt
except:
  pass

import numpy as np


def gan_grid_plot(samples, show=False, h=None, w=None,
                     wspace=0.03, hspace=0.03):
  # Check samples' shape to make sure they can be shown in pyplt
  if samples.shape[-1] == 1:
    samples = samples.reshape(samples.shape[:-1])

  # Plot samples
  sample_num = samples.shape[0]
  h = int(np.ceil(np.sqrt(sample_num))) if h is None else h
  w = int(np.ceil(sample_num / h)) if w is None else w

  fig = plt.figure(figsize=(w, h))
  gs = gridspec.GridSpec(h, w)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample, cmap='Greys_r')

  plt.tight_layout()
  fig.subplots_adjust(wspace=wspace, hspace=hspace)

  if show:
    plt.show()

  return fig


def save_plt(fig, filename):
  plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
  plt.close(fig)
