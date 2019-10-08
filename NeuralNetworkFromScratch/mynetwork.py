import numpy as np
from random import random
import pickle,gzip
import json
import sys
import matplotlib.pyplot as plt
import time

def vectorize(y):
    vc = np.zeros((10,1))
    vc[y] = 1.0
    return vc

def load_raw():
    f = gzip.open("../data/mnist.pkl.gz","rb")
    train,valid,test = pickle.load(f,encoding='bytes')
    f.close()
    return(train,valid,test)

def load_data():
    tr,vd,ts = load_raw()
    
    train_inp = [np.reshape(x,(784,1)) for x in tr[0]]
    train_res = [vectorize(y) for y in tr[1]]

    valid_inp = [np.reshape(x,(784,1)) for x in vd[0]]
    valid_res = [vectorize(y) for y in vd[1]]

    test_inp = [np.reshape(x, (784,1)) for x in ts[0]]
    test_res = [vectorize(y) for y in ts[1]]

    return (list(zip(train_inp, train_res)),list(zip(valid_inp,valid_res)),list(zip(test_inp,test_res)))

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def plot(*lst):
    for x in lst:
        plt.plot(x)
    plt.legend()
    plt.show()





class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return a-y
    

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)










class MyNetwork(object):
    
    def default_weight_initializer(self):
        self.bias_vectors = [np.random.randn(y,1) for y in self.nn_layers_sizes[1:]]
        
        
        self.weights_vectors = [np.random.randn(nxtNode,prevNode)/np.sqrt(prevNode) 
        for nxtNode, prevNode in zip(self.nn_layers_sizes[1:],self.nn_layers_sizes[:-1])]
    


    def large_weight_initializer(self):
        self.bias_vectors = [np.random.randn(y, 1) for y in self.nn_layers_sizes[1:]]
        self.weights_vectors = [np.random.randn(nxtNode, prevNode) for nxtNode, prevNode in zip(self.nn_layers_sizes[1:], self.nn_layers_sizes[:-1])]
    


    def __init__(self, sizes, cost = CrossEntropyCost):
        self.nn_layers_sizes = sizes
        self.nn_num_layers = len(sizes)
        ## INITIALIZE WEIGHTS
        self.default_weight_initializer()
        self.cost = cost
        


    def feedforward(self,inputs):
        for (bias, wgts) in zip(self.bias_vectors, self.weights_vectors):
            inputs = sigmoid(np.dot(wgts,inputs)+bias)
            # print("ASDF")
        return inputs 
    
    def backprop(self,input, output, full_matrix_based=False):
        nabla_b = [np.zeros(b.shape) for b in self.bias_vectors]
        nabla_w = [np.zeros(w.shape) for w in self.weights_vectors]

        current_activation = input
        activations = [current_activation]
        zs = []

        ##FORWARD PASS##

        for b,w in zip(self.bias_vectors,self.weights_vectors):
            z = np.dot(w,current_activation)+b
            zs.append(z)
            current_activation = sigmoid(z)
            activations.append(current_activation)
        
        ##BACKWARD PASS##
        delta_L = self.cost.delta(zs[-1], activations[-1], output)

        if full_matrix_based:
            nabla_b[-1] = np.sum(delta_L, axis=1, keepdims=True)
        else:
            nabla_b[-1] = delta_L
        nabla_w[-1] = np.dot(delta_L, activations[-2].transpose())
        # Here we need to transpose the activations in the previous layer because of the following reason --
        # delta vector is from the next layer, so it has a dimension of (l+1)X1, while 
        # previous layer's activation vector has the dimension of lx1, the weight vector 
        # between these two layers have the dimension of (l+1)xl, and we need to align 
        # delta vector the same way, so we transpose the activations

        for l in range(2,self.nn_num_layers):
            delta_l = np.dot(self.weights_vectors[-l+1].transpose(), nabla_b[-l+1]) * sigmoid_prime(zs[-l])
            if full_matrix_based:
                nabla_b[-l] = np.sum(delta_l, axis=1, keepdims=True)
            else:
                nabla_b[-l] = delta_l
            nabla_w[-l] = np.dot(delta_l, activations[-l-1].transpose())

        return (nabla_b,nabla_w)

    
    def update_mini_batch(self, minibatch,lr, lmbda,n,full_matrix_based=False, lmbda_type="L2"):
        minibatchlen = len(minibatch)
        acc_nabla_b = [np.zeros(b.shape) for b in self.bias_vectors]
        acc_nabla_w = [np.zeros(w.shape) for w in self.weights_vectors]
        if full_matrix_based:
            x, y = zip(*minibatch)
            x = np.hstack(x)
            y = np.hstack(y)
            acc_nabla_b, acc_nabla_w = self.backprop(x,y,full_matrix_based = True)
        else:
            for x,y in minibatch:
                del_nabla_b, del_nabla_w = self.backprop(x, y)
                acc_nabla_b = [anb+dnb for anb,dnb in zip(acc_nabla_b, del_nabla_b)]
                acc_nabla_w = [anw+dnw for anw,dnw in zip(acc_nabla_w, del_nabla_w)]
            
        self.bias_vectors = [b-(lr/minibatchlen)*anb for b, anb in zip(self.bias_vectors, acc_nabla_b)]

        ## Regularization 
        if lmbda_type=="L2":
            self.weights_vectors = [(1-lr*(lmbda/n))*w-(lr/minibatchlen)*anw for w, anw in zip(self.weights_vectors, acc_nabla_w)]
        elif lmbda_type == "L1":
            self.weights_vectors = [w - lr*(lmbda/n)*np.sign(w) - (lr/minibatchlen)*anw for w, anw in zip(self.weights_vectors, acc_nabla_w)]

    def evaluate(self,data):
        test_results = [(np.argmax(self.feedforward(x)),y) for x,y in data]
        return sum(int(x==y) for x,y in test_results)

    def cost_function_derivative(self, output_activations, y):
        return output_activations-y




    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0

        X, Y = zip(*data)
        x = np.hstack(X)
        y = np.hstack(Y)

        # for yy in y:
        #     a = self.feedforward(x)
        #     if convert: 
        #         y = vectorize(y)
        #     cost+= self.cost.fn(a, y)/len(data)
        a = self.feedforward(x)
        cost+=self.cost.fn(a,y)/len(data)

        # Add Regularization
        cost+= 0.5*((lmbda/(len(data)))*sum(np.linalg.norm(w)**2 for w in self.weights_vectors))

        return cost
    


    def accuracy(self, data, convert=False):

        # if convert:
        results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for x,y in data]
        # else:
        #     results = [(np.argmax(self.feedforward(x)), y) for x,y in data]
        
        return sum(int(x==y) for (x,y) in results)
    




    def SGD(self, train_data, epochs, batch_size, learning_rate, 
    lmbda=0.0,
    full_matrix_based = False,
    evaluation_data=None,
    monitor_evaluation_cost=False,
    monitor_evaluation_accuracy=False,
    monitor_training_cost=False,
    monitor_training_accuracy=False
    ):
        if evaluation_data is not None:
            evaluation_data_size = len(evaluation_data)
        train_data_size = len(train_data)

        evaluation_costs, evaluation_accuracys = [], []
        training_costs, training_accuracys = [], []

        curTime = time.time() 

        for e in range(epochs):
            #shuffle train data
            np.random.shuffle(train_data)
            
            minibatches = [train_data[k:k+batch_size] for k in range(0,train_data_size, batch_size)]


            print("Running Epoch: {}/{}".format(e+1, epochs))
            curTime = time.time()
            for minibatch in minibatches:
                self.update_mini_batch(minibatch,learning_rate, lmbda, train_data_size, full_matrix_based=full_matrix_based)            
            
            print("Epoch {} done; Runtime -- {} second(s)".format(e,(time.time()-curTime)))
            
            if monitor_training_cost:
                cost = self.total_cost(train_data, lmbda)
                training_costs.append(cost)
                print("Cost on training data: {}".format(cost))
            
            if monitor_training_accuracy:
                accuracy = self.accuracy(train_data, convert=True)
                training_accuracys.append(100.0*accuracy/len(train_data))
                print("Accuracy on training data: {}/{}".format(accuracy,train_data_size))
            
            if monitor_evaluation_cost:
                eval_cost = self.total_cost(evaluation_data,lmbda,convert=True)
                evaluation_costs.append(eval_cost)
                print("Evaluation Cost: {}".format(eval_cost))


            if monitor_evaluation_accuracy:
                eval_acc = self.accuracy(evaluation_data)
                evaluation_accuracys.append(100.0*eval_acc/len(evaluation_data))
                print("Evaluation Accuracy: {}/{}".format(eval_acc,evaluation_data_size))
    
        return evaluation_costs, evaluation_accuracys, training_costs, training_accuracys

 
    def save(self, filename):
        data = {"sizes":self.nn_layers_sizes,
                "weights":[w.tolist() for w in self.weights_vectors],
                "biases":[b.tolist() for b in self.bias_vectors],
                "cost":str(self.cost.__name__)
        }

        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    

def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()

    cost = getattr(sys.modules[__name__], data["cost"])

    net = MyNetwork(data["sizes"], cost=cost)
    net.weights_vectors = [np.array(w) for w in data["weights"]]
    net.bias_vectors = [np.array(b) for b in data["biases"]]

    return net



def plot(arr):
    plt.plot(arr)
    plt.show()