import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import register_cmap

color_values = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
color_names = ["black", "blue", "lime", "red", "yellow", "white"]
color_map = list(zip(color_values, color_names))
ls_cmap = LinearSegmentedColormap.from_list('ls_cmap', color_map)
ls_cmap.set_bad((0, 0, 0))
register_cmap(cmap=ls_cmap)