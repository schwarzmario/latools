from functools import partial
from typing import Any
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.colors
from collections import defaultdict
from matplotlib import pyplot as plt
from abc import ABC
import numpy as np
import awkward as ak

class AxesHolder(ABC):
    def __init__(self, fig: matplotlib.figure.Figure | None = None, ax: matplotlib.axes.Axes | None = None):
        self.ax = ax
        self.fig = fig
    def _touch_fig_ax(self, fig: matplotlib.figure.Figure | None = None, ax: matplotlib.axes.Axes | None = None):
        """Three ways to get fig & ax, listed in order of precedence:
        1: passed to this method,
        2: passed to init
        3: created from scratch
        Note: fig might be None if ax was provided but not fig."""
        if ax is None:
            if self.ax is not None:
                ax = self.ax
                fig = self.fig
            else:
                fig, ax = plt.subplots()
        return fig, ax
class DrawablePlot(AxesHolder):
    def __init__(self, fig: matplotlib.figure.Figure | None = None, ax: matplotlib.axes.Axes | None = None):
        super().__init__(fig, ax)
    # More utilities to come...


class HistogramTask:
    def __init__(self, min: float, max: float, nbins: int = 1000, fcn = lambda x: x[0], 
                 min_entries_required: int | None = None, logy: bool = False, ax=None, 
                 label: str | None = None):
        self.min = min
        self.max = max
        self.nbins = nbins
        self.fcn = fcn
        self.min_entries_required = min_entries_required
        self.logy = logy
        self.ax = ax
        self.label=label
    def initialize(self):
        self.hist, self.edges = np.histogram([], bins=self.nbins, range=(self.min, self.max))
        self.nr_entries = 0
    def __call__(self, x, _):
        val = self.fcn(x)
        if val.ndim != 1:
            raise RuntimeError(f"array has to be 1-dim. Got {val.ndim}")
        val = val.to_numpy()
        hist, self.edges = np.histogram(val, bins=self.nbins, range=(self.min, self.max))
        self.hist += hist
        self.nr_entries += len(val)
        if self.min_entries_required is not None and (self.nr_entries >= self.min_entries_required):
            return True
        return False
    def finalize(self):
        self.draw()
    def draw(self, *, ax=None, **kwargs):
        if ax is None:
            if self.ax is not None:
                ax = self.ax
            else:
                _, ax = plt.subplots()
        if self.logy:
            ax.set_yscale("log")
        #target = plt if ax is None else ax
        ax.stairs(self.hist, self.edges, label=self.label, **kwargs)

class Histogram2DTask:    
    def __init__(self, x_min: float, x_max: float, x_nbins: int, y_min: float, y_max: float, y_nbins: int,
                 x_fcn = lambda x: x[0], y_fcn = lambda x: x[1],
                 min_entries_required: int | None = None, fig=None, ax=None, logz:bool = False, autocrop_input_arrays:bool=False):
        self.x_min = x_min
        self.x_max = x_max
        self.x_nbins = x_nbins
        self.y_min = y_min
        self.y_max = y_max
        self.y_nbins = y_nbins
        self.x_fcn = x_fcn
        self.y_fcn = y_fcn
        self.min_entries_required = min_entries_required
        self.fig = fig
        self.ax = ax
        self.logz = logz
        self.autocrop_input_arrays = autocrop_input_arrays
    def initialize(self):
        self.hist, self.x_edges, self.y_edges = np.histogram2d([], [], bins=[self.x_nbins, self.y_nbins], 
                                                               range=([[self.x_min, self.x_max],[self.y_min, self.y_max]]))
        self.nr_entries = 0
    def __call__(self, x, _):
        x_val = self.x_fcn(x)
        y_val = self.y_fcn(x)
        if x_val.ndim != 1 or y_val.ndim != 1:
            raise RuntimeError(f"array has to be 1-dim. Got {x_val.ndim} / {y_val.ndim}")
        if len(x_val) != len(y_val):
            if not self.autocrop_input_arrays:
                raise RuntimeError(f"x and y must have same length! have: {len(x_val)}, {len(y_val)}")
            ## HACK: since I see the files of the evt tier having less events as raw, dsp, I need to cut
            if len(x_val) > len(y_val):
                x_val = x_val[:len(y_val)]
            else:
                y_val = y_val[:len(x_val)]
        hist, self.x_edges, self.y_edges = np.histogram2d(x_val.to_numpy(), y_val.to_numpy(), bins=[self.x_nbins, self.y_nbins], 
                                                            range=([[self.x_min, self.x_max],[self.y_min, self.y_max]]))
        self.hist += hist
        self.nr_entries += len(x_val)
        if self.min_entries_required is not None and (self.nr_entries >= self.min_entries_required):
            return True
        return False
    def finalize(self):
        self.draw()
    def draw(self):
        if self.ax is not None:
            ax = self.ax
            fig = self.fig
        else:
            fig, ax = plt.subplots()
        norm = matplotlib.colors.LogNorm() if self.logz else None
        ret = ax.pcolor(self.x_edges, self.y_edges, self.hist.T, norm=norm)
        if fig is not None:
            fig.colorbar(ret, ax=ax)

class CategoricalHistogramTask(DrawablePlot):
    def __init__(self, fcn, *, keymap_fcn = None, sort=True, min_entries_required = None, ax = None):
        """fcn: transforms input arrays into a 1-dim array of categories"""
        super().__init__(None, ax)
        self.fcn = fcn
        self.keymap_fcn = keymap_fcn
        self.sort=sort
        self.min_entries_required = min_entries_required
    def initialize(self):
        self.cats_dict = defaultdict(int) # categories found and their frequencies
        self.nr_entries = 0
    def __call__(self, x, _):
        cats_arr = self.fcn(x)
        if cats_arr.ndim != 1:
            raise RuntimeError(f"array has to be 1-dim. Got {cats_arr.ndim}")
        cats, counts = np.unique(cats_arr.to_numpy(), return_counts=True)
        for cat, count in zip(cats, counts):
            self.cats_dict[cat] += count
        self.nr_entries += np.sum(counts)
        if self.min_entries_required is not None and (self.nr_entries >= self.min_entries_required):
            return True
        return False
    def finalize(self):
        if self.keymap_fcn:
            self._map_keys()
        #print(self.cats_dict)
        if self.sort:
            self.cats_dict = dict(sorted(self.cats_dict.items()))
        self.draw()
    def _map_keys(self):
        assert self.keymap_fcn is not None
        newdict = {}
        for k, v in self.cats_dict.items():
            newdict[self.keymap_fcn(k)] = v
        self.cats_dict = newdict
    def draw(self):
        _, ax = self._touch_fig_ax()
        ax.bar(list(self.cats_dict.keys()), list(self.cats_dict.values()))
        ax.tick_params(axis='x', labelrotation=90)

class CategoricalHistogram2DTask(DrawablePlot):
    def __init__(self, x_fcn, y_fcn, *, mode: str = "normal", keymap_fcn = None, sort=True, 
                 min_entries_required = None, fig = None, ax = None, logz:bool = False):
        super().__init__(fig, ax)
        self.x_fcn = x_fcn
        self.y_fcn = y_fcn
        self.mode = mode
        self.keymap_fcn = keymap_fcn
        self.sort = sort
        self.min_entries_required = min_entries_required
        self.ax = ax
        self.logz = logz
    def initialize(self):
        # outer map: x axis, innter map: y axis
        self.cats_dict = defaultdict(partial(defaultdict, int)) # arg has to be factory, not object
        self.nr_entries = 0
    def __call__(self, x, _):
        req_dimensions = 1 if self.mode == "normal" else 2
        cats_x_arr = self.x_fcn(x)
        cats_y_arr = self.y_fcn(x)
        if cats_x_arr.ndim != req_dimensions or cats_y_arr.ndim != req_dimensions:
            raise RuntimeError(f"arrays each has to be {req_dimensions}-dim. Got {cats_x_arr.ndim} and {cats_y_arr.ndim}")
        if len(cats_x_arr) != len(cats_y_arr): # same as in Histogram2DTask
            #raise RuntimeError(f"x and y must have same length! have: {len(cats_x_arr)}, {len(cats_y_arr)}")
            ## HACK: since I see the files of the evt tier having less events as raw, dsp, I need to cut
            if len(cats_x_arr) > len(cats_y_arr):
                cats_x_arr = cats_x_arr[:len(cats_y_arr)]
            else:
                cats_y_arr = cats_y_arr[:len(cats_x_arr)]

        match(self.mode):
            case "normal":
                zipped_array = ak.zip([cats_x_arr, cats_y_arr])
            case "cartesian":
                zipped_array = ak.flatten(ak.cartesian([cats_x_arr, cats_y_arr], axis=-1), axis=-1)
            case _:
                raise RuntimeError("Unknown mode "+self.mode)
        
        cats, counts = np.unique(zipped_array, return_counts=True)
        for cat, count in zip(cats, counts):
            #print(cat.__repr__(), count.__repr__())
            self.cats_dict[cat[0]][cat[1]] += count
            self.nr_entries += count
        if self.min_entries_required is not None and (self.nr_entries >= self.min_entries_required):
            return True
        return False
    def finalize(self):
        if self.keymap_fcn:
            self._map_keys()
        #print(self.cats_dict)
        self.draw()
    def _map_keys(self):
        assert self.keymap_fcn is not None
        newdict = defaultdict(partial(defaultdict, int))
        for x_key, old_y_dict in self.cats_dict.items():
            y_dict = newdict[self.keymap_fcn(x_key)] # defaultdict creates new inner dict here
            for y_key, count in old_y_dict.items():
                y_dict[self.keymap_fcn(y_key)] = count
            #newdict[self.keymap_fcn(x_key)] = y_dict
        self.cats_dict = newdict
    def draw(self):
        with plt.rc_context({"figure.figsize": (14, 10)}):
            fig, ax = self._touch_fig_ax()
        x_labels = list(self.cats_dict.keys())
        y_labels = set()
        for y_dict in self.cats_dict.values():
            y_labels = y_labels.union(y_dict.keys())
        y_labels = list(y_labels) # cannot subscript or sort a set
        if self.sort:
            x_labels = sorted(x_labels)
            y_labels = sorted(y_labels)
        draw_array = np.ndarray((len(x_labels), len(y_labels))) # need to transpose when doing pcolor()
        for x in range(len(x_labels)):
            for y in range(len(y_labels)):
                draw_array[x][y] = self.cats_dict[x_labels[x]][y_labels[y]]
        norm = matplotlib.colors.LogNorm() if self.logz else None
        ret = ax.pcolor(range(len(x_labels)), range(len(y_labels)), draw_array.T, norm=norm)
        ax.set_xticks(range(len(x_labels)), x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_yticks(range(len(y_labels)), y_labels)    
        if fig is not None:
            fig.colorbar(ret, ax=ax)    