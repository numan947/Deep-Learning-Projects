import sys, collections, os, math, random, re, time, tarfile, zipfile
import numpy as np, matplotlib.pyplot as plt
from IPython import display
# d2l = sys.modules[__name__]


## PYTORCH

import 





def use_svg_display():
    """Use the svg format to display plot in jupyter."""
    display.set_matplotlib_formats("svg")

def set_figsize(figsize=(3.5, 2.5)):
    """Change default figure size -> (Width, Height)"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images. Must be converted to numpy."""
    figsize = (num_cols*scale, num_rows*scale)

    _, axes = plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])
    return axes

class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer and record the time in a list"""
        self.times.append(time.time() - self.start_time)
        return self.times[-1]
        
    def avg(self):
        """Return the average time"""
        return sum(self.times)/len(self.times)
    
    def sum(self):
        """Return the sum of time"""
        return sum(self.times)
        
    def cumsum(self):
        """Return the accumuated times"""
        return np.array(self.times).cumsum().tolist()

# Defined in file: ./chapter_linear-networks/linear-regression.md
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """A utility function to set matplotlib axes"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=None, figsize=(3.5,2.5),axes=None):
    """Plot multiple lines. X and Y must be numpy array -> X, Y both should be 2D array"""
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    
    ### What happened here???
    if not hasattr(X[0], "__len__"): X = [X]
    if Y is None: X, Y = [[]]*len(X), X
    if not hasattr(Y[0], "__len__"): Y = [Y]
    if len(X) != len(Y): X = X * len(Y)
    if not fmts: fmts = ['-']*len(X)
    axes.cla()

    for x, y, fmt in zip(X, Y, fmts):
        if(len(x)):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def synthetic_data(w, b, num_examples):
    """generate y = X w + b + noise using numpy array"""
    X = np.random.normal(scale=1.0,size=(num_examples, len(w)))
    y = np.dot(X,w)+b
    y+=np.random.normal(scale=0.01, size=y.shape)

    return X,y







    