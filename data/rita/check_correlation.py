from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
d = pd.read_csv('rita.content', delimiter='\t', header=None)
# del d[0] # remove the node indices
# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

f.canvas.draw()

xlabels = [str(int(item.get_text()) + 1) for item in ax.get_xticklabels()]
ax.set_xticklabels(xlabels)

ylabels = [str(int(item.get_text()) + 1) for item in ax.get_yticklabels()]
ax.set_yticklabels(ylabels)

plt.show()
