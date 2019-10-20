import sys, collections, os, math, random, re, time, tarfile, zipfile
import numpy as np, matplotlib.pyplot as plt
from IPython import display
# d2l = sys.modules[__name__]


## PYTORCH
import torch
import torch.utils.data as data

import torchvision
from torchvision.transforms import transforms





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


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a Pytorch data loader--> data_arrays must be (TensorX, TensorY)"""
    dataset = data.TensorDataset(*(data_arrays))
    return data.DataLoader(dataset,batch_size=batch_size, shuffle=is_train)


def get_dataloader_workers(num_workers=4):
    if sys.platform.startswith("win"):
        return 0
    else:
        return num_workers


def accuracy(y_hat, y):
    """Accuracy for Softmax Regression --> y_hat is one_hotted, y is not, both are Tensor, returns number of correctly classified examples"""
    return (y_hat.argmax(axis=1) == y).sum().item()


def evaluate_accuracy(net, data_iter, device=torch.device('cpu')):
    """Mainly for Softmax Regression"""
    net.eval() #change to evaluation mode
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device),0

    for X,y in data_iter:
        X,y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            acc_sum+=torch.sum((torch.argmax(net(X), dim=1) == y))
            n+=y.shape[0]
    return acc_sum.item()/n


def evaluate_loss(net, data_iter, criterion):
    """Evaluate the loss of a model on the given dataset"""

    metric = Accumulator(2)
    for X,y in data_iter:
        metric.add(criterion(net(X),y).sum().detach().numpy().item(), list(y.shape)[0])
    return metric[0]/metric[1]



class Accumulator(object):
    """Sum a list of numbers over time"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a+b for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0] * len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def train_ch3(net, train_iter, test_iter, criterion, num_epochs, batch_size, lr=None):
    """Train and evaluate a model with CPU"""
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        sum_loss, sum_acc, n = 0.0,0.0,0
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            y = y.type(torch.float32)
            sum_loss+=loss.item()
            sum_acc+=torch.sum((torch.argmax(y_hat, dim=1).type(torch.FloatTensor) == y).detach()).float()
            n+=list(y.size())[0]
        test_acc = evaluate_accuracy(net, test_iter)

        print("Epoch {}, loss {}, train acc {}, evaluation acc {}".format(epoch+1, sum_loss/n, sum_acc/n, test_acc))

#######################GPU TRY##########################


def try_gpu(i=0):
    if torch.cuda.is_available():
        return torch.device("cuda:"+str(i))
    return torch.device("cpu")

def try_all_gpus():
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            device = torch.device('cuda:'+str(i))
            devices.append(device)
    else:
        devices = [torch.device("cpu")]
    return devices

######################CNN####################
def corr2d(X, K):
    """X and K are Tensors"""
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = torch.sum(X[i:i+h, j:j+w]*K)
    return Y






##################### FASHION_MNIST###############################
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def load_data_fashion_mnist(batch_size, resize=None):
    tx = None
    if resize:
        tx = transforms.Compose([transforms.Resize(resize)])
    
    fashionTrain = torchvision.datasets.FashionMNIST("./data/FashionMnist", train=True, transform=tx, download=True)
    fashionTest = torchvision.datasets.FashionMNIST("./data/FashionMnist", train=False, transform=tx, download=True)

    trainLoader = data.DataLoader(fashionTrain, num_workers=get_dataloader_workers(4), batch_size=batch_size, shuffle=True)
    testLoader = data.DataLoader(fashionTest, num_workers=get_dataloader_workers(4), batch_size=batch_size, shuffle=False)

    return (trainLoader, testLoader)

