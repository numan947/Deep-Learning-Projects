import sys, collections, os, math, random, re, time, tarfile, zipfile
import numpy as np, matplotlib.pyplot as plt
from IPython import display
# d2l = sys.modules[__name__]


## PYTORCH
import torch
import torch.utils.data as data

import torchvision
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import datetime
import itertools


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


class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes,]
        # use a lambda to capture arguments
        self.config_axes = lambda : set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
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
    txs = []
    if resize:
        txs.append(transforms.Resize(resize))
    txs.append(transforms.ToTensor())
    tx = transforms.Compose(txs)
    
    fashionTrain = torchvision.datasets.FashionMNIST("./data/FashionMnist", train=True, transform=tx, download=True)
    fashionTest = torchvision.datasets.FashionMNIST("./data/FashionMnist", train=False, transform=tx, download=True)

    trainLoader = data.DataLoader(fashionTrain, num_workers=get_dataloader_workers(4), batch_size=batch_size, shuffle=True)
    testLoader = data.DataLoader(fashionTest, num_workers=get_dataloader_workers(4), batch_size=batch_size, shuffle=False)

    return (trainLoader, testLoader)

########################## My Trainers#############################################



def train_sgd(model, criterion, train_iter, test_iter, num_epochs, lr, device, cutoff=None, verbose=True, printBatch=False):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay=0.005)
    trainLosses, testLosses = [],[]
    trainAccs, testAccs = [],[]
    
    for e in range(num_epochs):
        currentBatch = 0
        curt = time.time()
        model.train()
        
        totalTrainLoss = torch.tensor(0.0, device=device)
        nTrain = torch.tensor(0.0, device=device)
        totalTrainCorrect = torch.tensor(0.0, device=device)

        for X, y in train_iter:
            currentBatch+=1

            optimizer.zero_grad()
            
            X,y = X.to(device), y.to(device)
            y_hat = model(X)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            totalTrainLoss+= loss*X.shape[0]
            nTrain+=X.shape[0]
            totalTrainCorrect += (torch.argmax(y_hat, dim=1) == y).sum()


            if(printBatch or verbose):
                sys.stdout.write("\r Epoch{}".format(e+1)+" "+str(currentBatch)+" / "+str(len(train_iter))) 
            if cutoff is not None and currentBatch>=cutoff:
                break               

        with torch.no_grad():
            
            averageTrainLoss = (1.0*totalTrainLoss/nTrain).item()
            averageAccuracy = (1.0*totalTrainCorrect/nTrain).item() 
            
            trainLosses.append(averageTrainLoss)
            trainAccs.append(averageAccuracy)


            model.eval()
            # totalTestLoss = 0.0
            # nTest = 0
            # totalTestCorrect = 0
            totalTestLoss = torch.tensor(0.0, device=device)
            nTest = torch.tensor(0.0, device=device)
            totalTestCorrect = torch.tensor(0.0, device=device)

            for X,y in test_iter:
                
                X,y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                totalTestLoss+=X.shape[0]*loss
                nTest+=X.shape[0]
                totalTestCorrect+=(torch.argmax(y_hat, dim=1)==y).sum()
                if cutoff is not None and currentBatch>=cutoff:
                    break       
            

            averageTestLoss = (1.0*totalTestLoss/nTest).item()
            averageTestAccuracy = (1.0*totalTestCorrect/nTest).item()
            
            testLosses.append(averageTestLoss)
            testAccs.append(averageTestAccuracy)
            
            if verbose:
                print("Epoch ", e+1)
                print("Average TrainLoss {0:.3f}".format(averageTrainLoss))
                print("Training Accuracy {0:.3f}".format(averageAccuracy))
            
                print("Average TestLoss {0:.3f}".format(averageTestLoss))
                print("Testset Accuracy {0:.3f}".format(averageTestAccuracy))
                print("Time Needed: ", datetime.timedelta(seconds=(time.time() - curt)))
    return trainLosses, trainAccs, testLosses, testAccs

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# import datetime
def generate_lr_report(train_function, net, criterion, train_iter, test_iter, num_epochs, device, lrs, cutoff=5):
    # print(int(np.ceil(len(lrs)/4.0)), int(np.floor(len(lrs)/4.0)))
    _, axes = plt.subplots(nrows=int(np.ceil(len(lrs)/4.0)), ncols=4, figsize=(12,8))
    axes = axes.flatten()
    axes = axes[:len(lrs)]

    for lr,ax in zip(lrs,axes):
        net.apply(weight_reset)
        curTime = time.time()
        trl,tra,tsl,tsa = train_function(net, criterion, train_iter, test_iter, num_epochs, lr, device, cutoff=cutoff, verbose=False,printBatch=True)
        print("Learning rate: ", lr)
        print("Train Losses: ", trl)
        print("Test Losses : ", tsl)
        print("Train Accuracies: ", tra)
        print("Test Accuracies : ", tsa)
        print("Time Needed: ", datetime.timedelta(seconds=time.time()-curTime))
        ax.plot(tra, label="train acc")
        ax.plot(tsa, label="test acc")
        ax.plot(trl, label="train loss")
        ax.plot(tsl, label="test loss")
        ax.title.set_text("Learning Rate: "+str(lr))
        ax.legend()
    return axes


    ####################TEXT PROCESSING#########################

def read_time_machine():
    """Load Time Machine book into a list of sentences"""
    with open("./timemachine.txt", "r") as f:
        lines = f.readlines()
    
    return [re.sub("[^A-Za-z]+",' ', line.strip().lower()) for line in lines]

def tokenize(lines, token="word"):
    if token == "word":
        return [line.split(' ') for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print("Unknown Token Type")


def count_corpus(sentences):
    tokens = [toks for line in sentences for toks in line]
    return collections.Counter(tokens)

class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x : x[0])
        self.token_freqs.sort(key=lambda x:x[1], reverse=True)

        if use_special_tokens:
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            uniq_tokens = ['<pad>','<bos>','<eos>','<unk>']
        
        else:
            self.unk, uniq_tokens = 0, ['<unk>']
        
        uniq_tokens.extend([token for token, freq in self.token_freqs if freq>=min_freq and token not in uniq_tokens])

        self.idx_to_token, self.token_to_idx = [], dict()

        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token)-1
    
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_token(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[tk] for line in tokens for tk in line]
    if max_tokens>0: 
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Each sequence in the batches starts at random"""
    corpus = corpus[random.randint(0, num_steps):]
    num_examples = ((len(corpus)-1)//num_steps)
    example_indices = list(range(0, num_examples*num_steps, num_steps))
    random.shuffle(example_indices)
    data = lambda pos: corpus[pos:pos+num_steps] ## Function, given index j, returns jth slice
    
    num_batches = num_examples//batch_size

    for i in range(0, batch_size*num_batches, batch_size):
        batch_indices = example_indices[i:i+batch_size]
        X = [data(j) for j in batch_indices]
        Y = [data(j+1) for j in batch_indices]

        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_consecutive(corpus, batch_size, num_steps):
    """Each sequence in the batch follows after the sequence in the previous batch"""
    offset = random.randint(0, num_steps)
    num_indices = ((len(corpus) - offset - 1)//batch_size) * batch_size

    Xs = torch.tensor(corpus[offset:offset+num_indices])
    Ys = torch.tensor(corpus[offset+1: offset+1+num_indices])
    
    Xs, Ys = Xs.view((batch_size, -1)), Ys.view((batch_size, -1))
    # print(Xs)
    # print(Ys)
    num_batches = Xs.shape[1]//num_steps
    for i in range(0, num_batches*num_steps, num_steps):
        X = Xs[:, i:(i+num_steps)]
        Y = Ys[:, i:(i+num_steps)]
        yield X, Y

class SeqDataLoader(object):
    """Sequence Loader Object"""
    def __init__(self, corpus, vocab, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            data_iter_fn = seq_data_iter_random
        else:
            data_iter_fn = seq_data_iter_consecutive
        
        self.corpus, self.vocab = corpus, vocab
        self.get_iter = lambda: data_iter_fn(self.corpus, batch_size, num_steps)

    def __iter__(self):
        return self.get_iter()


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=1000):
    data_iter = SeqDataLoader(*(load_corpus_time_machine()), batch_size, 
                                num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

def grad_clipping(params, theta, ctx):
    norm = torch.Tensor([0], device=ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data.mul_(theta / norm)


def generate(prefix, num_predicts, model, vocab, device):
    state = model.init_hidden(batch_size=1, device=device).float()
    model = model.to(device)
    outputs = [vocab[prefix[0]]]

    get_input = lambda: F.one_hot(torch.tensor(outputs[-1], device=device).view(1,1), len(vocab)).float()

    for y in prefix[1:]:
        # print(state.shape)
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    
    for _ in range(num_predicts):
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(dim=1).item()))
    # print(outputs)
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def train_recurrent_model(model, criterion, optimizer, vocab, train_iter,  batch_size, num_epochs, device, clip_val=5.0, batch_first=True):
    
    model = model.to(device)
    train_losses = []
    perplexities = []

    for e in range(num_epochs):
        h = model.init_hidden(batch_size,device)
        total_loss = 0
        total_examples = 0
        for X, y in train_iter:            
            if batch_first:
                X,y = torch.nn.functional.one_hot(X, len(vocab)).to(device).float(),y.to(device).flatten().float()
            else:
                X,y = torch.nn.functional.one_hot(X.T, len(vocab)).to(device).float(),y.T.reshape(-1,).to(device).float()

            if model.recurrent_type=="lstm":
                h = tuple(h.detach() for h in h)
            else:
                h = h.detach()
            optimizer.zero_grad()

            out,h = model(X, h)

            loss = criterion(out, y.long())

            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(),clip_val)

            optimizer.step()
            total_loss+= loss.item()*X.shape[0]
            total_examples+=X.shape[0]
        
        perplexities.append(np.exp(total_loss/total_examples))
        train_losses.append(total_loss/total_examples)
        print("Epoch {}  Loss {} perplexity {}".format(e, train_losses[-1], perplexities[-1]))
    
    return train_losses, perplexities



########################Neural Machine Translation####################################
def read_data_nmt(name):
    with open(name, 'r') as f:
        return f.read()

def preprocess_nmt(text):
    text = text.replace("\u202f"," ").replace("\xa0"," ")

    space_needed = lambda char, prev_char: True if char in (',','!','?','.') and prev_char !=' ' else False
    out = [" "+char if i>0 and space_needed(char, text[i-1]) else char for i, char in enumerate(text.lower())]

    return ''.join(out) 

def tokenize_nmt(text, num_examples=None):
    source, target = [], []

    for i, line in enumerate(text.split("\n")):
        if num_examples and i > num_examples:
            break
        parts = line.split("\t")

        if(len(parts)==2):
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    return source, target

def trim_pad(line, num_steps, padding_token):
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[padding_token]*(num_steps-len(line))


def build_array(lines, vocab, num_steps, is_source):
    lines = [vocab[l] for l in lines]

    if not is_source:
        lines = [[vocab.bos]+l+[vocab.eos] for l in lines]
    
    array = torch.tensor([trim_pad(l, num_steps, vocab.pad) for l in lines])
    valid_len = (array!=vocab.pad).sum(dim=1)

    return array, valid_len

def load_data_nmt(filename, batch_size, num_steps, num_examples=1000):
    text = preprocess_nmt(read_data_nmt(filename))
    source, target = tokenize_nmt(text, num_examples)
    
    src_vocab = Vocab(source, min_freq=2, use_special_tokens=True)
    tgt_vocab = Vocab(target, min_freq=2, use_special_tokens=True)

    source_array, source_valid_len = build_array(source, src_vocab, num_steps, is_source=True)
    target_array, target_valid_len = build_array(target, tgt_vocab, num_steps, is_source=False)

    data_arrays = (source_array, source_valid_len, target_array, target_valid_len)

    data_iter = load_array(data_arrays, batch_size)

    return src_vocab, tgt_vocab, data_iter




class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
    
    def forward(self, X):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def forward(self,X, state):
        raise NotImplementedError
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):

        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)

        return self.decoder(dec_X, dec_state)




def SequenceMask(X, X_len, value=0):
    all_masks = []

    for sen,vlen in zip(X,X_len):
        tmp_m = []
        # print(sen,vlen)
        for i in range(len(sen)):
            if(i<vlen):
                tmp_m.append(True)
            else:
                tmp_m.append(False)
        all_masks.append(tmp_m)
    
    X[~(torch.tensor(all_masks))] = value
    return X


class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, preds, labels, valid_length):
        weights = torch.ones_like(labels)
        weights = SequenceMask(weights, valid_length).float()
        self.reduction="none"

        output = super(MaskedCrossEntropyLoss, self).forward(preds.reshape(-1,preds.shape[-1]), labels.flatten())
        return (output.view(weights.shape)*weights).mean(dim=1)


## X: a 3D tensor, valid_length: 2D or 1D tensor --> batchwise or individual
def masked_softmax(X, valid_length=None):
    softmax = nn.Softmax(dim=1)
    if valid_length is None:
        return softmax(X)
    else:
        device = X.device
        X = X.cpu()
        valid_length = valid_length.cpu()
        shape = X.shape
        if valid_length.ndim == 1:
            valid_length = torch.FloatTensor(valid_length.numpy().repeat(shape[1], axis=0))
        else:
            valid_length = valid_length.flatten()
        X = SequenceMask(X.view((-1, shape[-1])), valid_length, value=-1e6) # large negative numbers' exp is zero, that is why value=-1e6

        return softmax(X).reshape(shape).to(device)

class MLPAttention(nn.Module):
    ## key_dim is the dimensions of individual keys and values
    def __init__(self, key_dim, hidden_units, dropout):
        super(MLPAttention, self).__init__()

        self.W_k = nn.Linear(key_dim, hidden_units, bias=False)
        self.W_q = nn.Linear(key_dim, hidden_units, bias=False)
        self.v = nn.Linear(hidden_units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length=None):
        query, key = torch.tanh(self.W_q(query)), torch.tanh(self.W_k(key))
        features = query.unsqueeze(dim=2)+key.unsqueeze(1)
        scores = self.v(features).squeeze(-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))

        return torch.bmm(attention_weights, value)


def train_encoder_decoder_model(model, data_iter, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    model = model.apply(weight_reset)

    animator = Animator(xlabel="epoch", ylabel="loss", xlim=[1, num_epochs])

    for e in range(1, num_epochs+1):
        timer = Timer()
        metric = Accumulator(2)
        # sys.stdout.write("\rRunning, {}".format(e))

        for X, X_vlen, Y, Y_vlen in data_iter:
            X, X_vlen, Y, Y_vlen = X.to(device), X_vlen.to(device), Y.to(device), Y_vlen.to(device)
            Y_input, Y_label, Y_vlen = Y[:,:-1], Y[:, 1:], Y_vlen-1

            optimizer.zero_grad()

            Y_hat, state = model(X, Y_input, X_vlen, Y_vlen)
            loss = criterion(Y_hat, Y_label, Y_vlen).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            num_tokens = Y_vlen.sum().item()
            metric.add(loss.item(), num_tokens)
        if e%10==0:
            animator.add(e, metric[0]/metric[1])
    print("loss {}, {} tokens/sec in {}".format(metric[0]/metric[1], metric[1]/timer.stop(), device))

def predict_seq2seq(model, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """Input one sentece at a time"""
    model = model.to(device)
    src_tokens = src_vocab[src_sentence.lower().split(" ")]
    enc_valid_length = torch.tensor([len(src_tokens)], device=device, dtype=torch.float32)
    src_tokens = trim_pad(src_tokens, num_steps, src_vocab.pad)

    enc_X = torch.tensor(src_tokens, device=device).float()

    enc_outputs = model.encoder(enc_X.unsqueeze(dim=0), enc_valid_length)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_length)
    dec_X = torch.tensor([tgt_vocab.bos], device=device, dtype=torch.float32).unsqueeze(dim=0)

    predicted_tokens = []

    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        dec_X = torch.argmax(Y, dim=2)

        py = dec_X.squeeze(dim=0).int().item()

        if py == tgt_vocab.eos:
            break
        predicted_tokens.append(py)
    
    return ' '.join(tgt_vocab.to_token(predicted_tokens))




##########################COMPUTER VISION####################################

## Supports batching of images (batchsize, channel, height, width)
def MultiBoxPrior(Images, sizes, ratios):
    img_boxes = []
    for Image in Images:
        H,W = Image.shape[-2], Image.shape[-1]
        step_x = 1.0/W
        step_y = 1.0/H
        boxes = []
        for h,w in itertools.product(range(H), range(W)):
            # centering in the pixel
            cx = (w+1)*step_x
            cy = (h+1)*step_y
            s1 = sizes[0]
            r1 = ratios[0]

            total_width = 2.2*w*s1*math.sqrt(r1)*step_x
            total_height = 2.25*(h*s1*step_y)/math.sqrt(r1)
            boxes.append(
                (cx-total_width/2.0, cy-total_height/2.0, cx+total_width/2.0, cy+total_height/2.0)
                )
            
            for s, r in zip(sizes[1:],ratios[1:]):
                ## giving the height and width some scaling to match MXNet implementation
                total_width1 = 2.2*w*s1*math.sqrt(r)*step_x
                total_height1 = 2.25*(h*s1*step_y)/math.sqrt(r)
                ## giving the height and width some scaling to match MXNet implementation
                total_width2 = 2.2*w*s*math.sqrt(r1)*step_x
                total_height2 = 2.25*(h*s*step_y)/math.sqrt(r1)
                boxes.extend(
                    [(cx-total_width1/2.0, cy-total_height1/2.0, cx+total_width1/2.0, cy+total_height1/2.0),
                    (cx-total_width2/2.0, cy-total_height2/2.0, cx+total_width2/2.0, cy+total_height2/2.0)]
                    )
        img_boxes.append(boxes)
    return torch.Tensor(img_boxes)




def bbox_to_rect(bbox, color):
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], fill=False, edgecolor=color, linewidth=2.0
    )

def show_bboxes(axes, bboxes, labels=None, colors=None):
    def __make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        
        return obj
    
    labels = __make_list(labels)
    colors = __make_list(colors, ['b','g','r','m','c'])

    for i, bbox in enumerate(bboxes):
        color = colors[i%len(colors)]
        rect = bbox_to_rect(bbox.numpy(), color)
        
        axes.add_patch(rect)

        if labels and len(labels)>i:
            text_color='k' if color =='w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center',fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))