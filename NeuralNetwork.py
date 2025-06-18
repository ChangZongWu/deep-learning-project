import sklearn
from sklearn.datasets import make_circles, make_moons
import torch
from sympy.codegen import Print
from timm.utils import random_seed
from  torch import nn # nn contains all of pytorch's building block for neural networks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import device
from pathlib import Path
from sklearn.model_selection import  train_test_split
from torch.utils.benchmark.utils.fuzzer import dtype_size

from introTensor import tensor_back_on_cpu, RANDOM_SEED, tensor_A
from workflow import x_train, y_train, y_test, loss_fn, optimizer, weight

# 02 Neural Network classification with Pytorch
# classification is a problem of predicting whether something is one thing or another (can be multi things)

# 1. Make classification data
# make 1000 samples
n_samples = 1000

# create circles
x, y = make_circles(n_samples,noise=0.03,random_state=42)
print(len(x),len(y))

# make dataframe of circle data
circles = pd.DataFrame({"X1":x[:,0],
                        "X2":x[:,1],
                        "label":y})
print(circles.head(10))

# plot the
# plt.scatter(x=x[:,0],y=x[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()

# the data we are working with is often referred to as a toy dataset, a dataset that is small enough
# to experiment but still sizeable enough to practice the fundamentals.

# 1.1 check input and output shapes

print(x.shape)
print(y.shape)

x_sample = x[0]
y_sample = y[0]
print(x_sample,y_sample)
print(x_sample.shape,y_sample.shape)

# 1.2 turn data into tensors and create train and test splits
# turn data into tensors
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(x[:5],y[:5])

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,# 0.2 = 20% of data will be tested 80% be trained
                                                    random_state=42)
print(len(X_train),len(X_test),len(y_train),len(y_test))

# build a model
# 1. setup device
# 2. construct a model
# 3. define a loss function and optim
# 4. create training and test loop

# 1
device = "cuda" if torch.cuda.is_available() else "cpu"
# 2
# 2.1 subclasses nn.Module
# 2.2 create 2 nn.Linear layers to hand the shapes of our data
# 2.3 define forward method that outlines the forward pass of the model
# 2.4 instatiate and instance of our model class and send it to the target device

# 2.1
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2.2
        self.layer_1 = nn.Linear(in_features=2, out_features= 5) #takes in 2 features and upscales to 5
        self.layer_2 = nn.Linear(in_features=5, out_features= 1) #takes in 5 features from previous layer and outputs a single feature
    # 2.3
    def forward(self,x):
        return self.layer_2(self.layer_1(x)) # x -> layer1 -> layer2 -> output

# 2.4
model_0 = CircleModelV0().to(device)

# replicate the model above using nn.sequential
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5,out_features=1)
).to(device)

print(model_0)

# make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(len(untrained_preds))
print(untrained_preds.shape)

# 3
# 3.1 set up loss func and optim
#loss_fn = nn.BCELoss() requires input to have gone through the sigmoid activation function prior to input to BCEloss
loss_fn = nn.BCEWithLogitsLoss() #sigmoid activation function built-in

optimizer =torch.optim.SGD(params=model_0.parameters(),lr=0.1)
# calculate accuracy
def accuacy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

# 4 training model
# training loop
# 1.forward pass
# 2.calculate the loss
# 3.optimizer zero grad
# 4.loss backward
# 5. optimizer step

model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)

# use the sigmoid activation function on our model logits to turm them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

# find the predicted labels
y_preds = torch.round(y_pred_probs)

# in full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# check equality
print(torch.eq(y_preds.squeeze(),y_pred_labels.squeeze()))

# get rid of extra dimension
print(y_preds.squeeze())

# 4.2 build a training and test loop
torch.manual_seed(42)

# Set the number of epochs
epochs = 10

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(
        X_train).squeeze()  # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
    y_pred = torch.round(torch.sigmoid(y_logits))  # turn logits -> pred probs -> pred labls

    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train)
    loss = loss_fn(y_logits,  # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train)
    acc = accuacy_fn(y_true=y_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuacy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    # if epoch % 10 == 0:
    #     print(
    #         f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# make a predictions and eval the model
import requests

if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# plot
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_0,X_train,y_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_0,X_test,y_test)
#plt.show()

# improving the model
# 1 add more layers - give the model more chances to learn about the patterns in the data
# 2 add more hidden units - go from 5 hidden units to 10 hidden units
# 3 fit for longer
# 4 changing the activation function
# 5 change the learning rate
# 6 change the loss function

class CircleModelV1(nn.Module):
     def __init__(self):
         super().__init__()
         self.layer_1 = nn.Linear(in_features= 2, out_features= 10)
         self.layer_2 = nn.Linear(in_features= 10, out_features= 10)
         self.layer_3 = nn.Linear(in_features= 10,out_features= 1)
     def forward(self,x):
         # z = self.layer_1(x)
         # z = self.layer_2(z)
         # z = self.layer_3(z)
         return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)
print(model_1)
# create a loss function
loss_fn = nn.BCEWithLogitsLoss()
# create a optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.1)
# write a training and evaluation loop for model_1
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
for epoch in range(epochs):
    model_1.train()

    y_logits= model_1(X_train).squeeze()

    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,y_train)
    acc = accuacy_fn(y_true=y_train,y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuacy_fn(y_true=y_test,y_pred=test_pred)
    # if epoch %100 ==0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc: .2f}% | Test loss: {test_loss}:.5f, Test acc: {test_acc:.2f}%")

# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_0,X_train,y_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_0,X_test,y_test)
#plt.show()

# preparing data to see if our data fita straight line

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

x_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * x_regression + bias

train_split = int(0.8*len(x_regression))
x_regression_train , y_regression_train = x_regression[:train_split],y_regression[:train_split]
x_regression_test, y_regression_test = x_regression[train_split:],y_regression[train_split:]

# plot_predictions(train_data=x_regression_train,
#                  train_labels=y_regression_train,
#                  test_data=x_regression_test,
#                  test_labels=y_regression_test)
#plt.show()

# adjust model_1 to fit a straight line
model_2 = nn.Sequential(
    nn.Linear(in_features=1,out_features=10),
    nn.Linear(in_features=10,out_features=10),
    nn.Linear(in_features=10,out_features=1)
).to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.01)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs=10
x_regression_train,y_regression_train = x_regression_train.to(device),y_regression_train.to(device)
x_regression_test,y_regression_test = x_regression_test.to(device), y_regression_test.to(device)

for epoch in range(epochs):
    y_pred = model_2(x_regression_train)
    loss = loss_fn(y_pred,y_regression_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(x_regression_test)
        test_loss = loss_fn(test_pred,y_regression_test)

    # if epoch % 100 ==0:
    #     print(f"epoch: {epoch}| loss: {loss:.5f} | test loss:{test_loss:.5f}")

# Turn on eval mode
model_2.eval()

# make predictions
with torch.inference_mode():
    y_preds = model_2(x_regression_test)
#plot_predictions(train_data=x_regression_train.cpu(),train_labels=y_regression_train.cpu(),
#                 test_data=x_regression_test.cpu(),test_labels=y_regression_test.cpu(),predictions=y_preds.cpu())
#plt.show()

# 6. non-linearity

#6.1
n_samples = 1000
x,y = make_circles(n_samples,noise=0.03,random_state=42)
#plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
#plt.show()

x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# 6.2

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,out_features=15)
        self.layer_2 = nn.Linear(in_features=15, out_features=15)
        self.layer_3 = nn.Linear(in_features=15,out_features=1)
        self.relu = nn.ReLU() # relu is a non-linear activation func
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

#6.3
torch.manual_seed(42)
torch.cuda.manual_seed(42)

x_train,y_train=x_train.to(device),y_train.to(device)
x_test, y_test = x_test.to(device),y_test.to(device)
epochs=1100
for epoch in range(epochs):
    model_3.train()
    y_logits=model_3(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,y_train)
    acc = accuacy_fn(y_true=y_train,y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuacy_fn(y_true=y_test,y_pred=test_pred)
    # if epoch %100 ==0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc: .2f}% | Test loss: {test_loss}:.4f, Test acc: {test_acc:.2f}%")

#6.4
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(x_test))).squeeze()


# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("train")
# plot_decision_boundary(model_3,x_train,y_train)
# plt.subplot(1,2,2)
# plt.title("test")
# plot_decision_boundary(model_3,x_test,y_test)
# plt.show()

#7 replicating non-linear activation functions

A = torch.arange(-10,10,1,dtype=torch.float32)
#plt.plot(A)
#plt.plot(torch.relu(A))

def relu(x: torch.Tensor) ->torch.Tensor:
    return torch.maximum(torch.tensor(0),x)

#plt.plot(relu(A))

def sigmoid(x):
    return 1/(1+torch.exp(-x))
#plt.plot(torch.sigmoid(A))
#plt.plot(sigmoid(A))

#8 putting all together multi class classification problem
#8.1 create a toy multi_class dataset
from sklearn.datasets import make_blobs
num_classes = 4
num_features = 2

#1 create multi-class data
x_blob, y_blob = make_blobs(n_samples=1000,n_features=num_features,centers=num_classes,cluster_std=1.5, # give the clusters a little shake up
                            random_state=42)
#2 turn data in to tensors
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
#3 split into train and test
x_blob_train,x_blob_test,y_blob_train,y_blob_test = train_test_split(x_blob,y_blob,
                                                                     test_size=0.2,random_state=42)
#4 plot
# plt.figure(figsize=(10,7))
# plt.scatter(x_blob[:,0],x_blob[:,1],c=y_blob,cmap=plt.cm.RdYlBu)
# plt.show()

# 8.2 build multi class classification
device = "cuda" if torch.cuda.is_available() else "cpu"
class BlobModel(nn.Module):
    def __init__(self, input_feature, output_feature, hidden_units = 8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(nn.Linear(in_features=input_feature,out_features=hidden_units),
                                                #nn.ReLU(),
                                                nn.Linear(in_features=hidden_units,out_features=hidden_units),
                                                #nn.ReLU(),
                                                nn.Linear(in_features=hidden_units,out_features=output_feature))
    def forward(self,x):
        return self.linear_layer_stack(x)

model_4 = BlobModel(input_feature=2,
                    output_feature=4,
                    hidden_units=8).to(device)

#8.3 loss func and optim
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(),lr=0.1)
#8.4 training loop
# model_4.eval()
with torch.inference_mode():
    y_logits = model_4(x_blob_train.to(device))

y_pred_probs = torch.softmax(y_logits,dim=1)
y_preds = torch.argmax(y_pred_probs,dim=1)
# 8.5 training loop and test loop

torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 100
x_blob_train,y_blob_train = x_blob_train.to(device),y_blob_train.to(device)
x_blob_test,y_blob_test = x_blob_test.to(device),y_blob_test.to(device)


for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(x_blob_train)
    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
    loss = loss_fn(y_logits,y_blob_train)
    acc = accuacy_fn(y_true=y_blob_train,y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(x_blob_test)
        test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits,y_blob_test)
        test_acc =accuacy_fn(y_true=y_blob_test,y_pred=test_preds)
    #if epoch %10 ==0:
        #print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc: .2f}% | Test loss: {test_loss}:.4f, Test acc: {test_acc:.2f}%")
#
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(x_blob_test)
y_pred_probs = torch.softmax(y_logits,dim=1)
y_preds = torch.argmax(y_pred_probs,dim=1)

#plot
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("train")
# plot_decision_boundary(model_4,x_blob_train,y_blob_train)
# plt.subplot(1,2,2)
# plt.title("test")
# plot_decision_boundary(model_4,x_blob_test,y_blob_test)
# plt.show()
#
from torchmetrics import Accuracy
#
# torch_acc = Accuracy.to(device)
# torch_acc(y_preds,y_blob_test)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
x,y = make_moons(n_samples=1000,noise=0.07,random_state=RANDOM_SEED)
print(x[:5])
print(y[:5])
from sklearn.datasets import make_moons
import pandas as pd
moons = pd.DataFrame({"X1":x[:,0],"X2":x[:,1],"label":y})
import matplotlib.pyplot as plt
# plt.scatter(x=x[:, 0],
#             y=x[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.show()
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=RANDOM_SEED)

import torch
from torch import nn
class MoonModelV0(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        super().__init__()
        self.layer_1 = nn.Linear(in_features= in_features,out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features= hidden_units,out_features=hidden_units)
        self.layer_3 = nn.Linear(in_features= hidden_units, out_features=out_features)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_0 = MoonModelV0(in_features=2,hidden_units=10,out_features=1).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(),lr=0.1)

x_train,y_train = x_train.to(device),y_train.to(device)
x_test, y_test = x_test.to(device),y_test.to(device)

y_logits = model_0(x_train).squeeze()
y_pred_probs = torch.sigmoid(y_logits)
y_pred = torch.round(y_pred_probs)

acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

torch.manual_seed(RANDOM_SEED)
epochs = 1000
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(x_train).squeeze()
    y_pred_probs = torch.sigmoid(y_logits)
    y_pred = torch.round(y_pred_probs)
    loss = loss_fn(y_logits,y_train)
    acc = acc_fn(y_pred,y_train.int())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logits=model_0(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,y_test)
        test_acc = acc_fn(test_pred,y_test.int())
    # if epoch % 100 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.2f} Acc: {acc:.2f} | Test loss: {test_loss:.2f} Test acc: {test_acc:.2f}")

# Plot the model predictions
import numpy as np


def plot_decision_boundary(model, X, y):
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_0,x_train,y_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_0,x_test,y_test)
# plt.show()

tensor_A = torch.arange(-100,100,1)
#plt.plot(tensor_A)
#plt.plot(torch.tanh(tensor_A))
def tanh(z):
    return (torch.exp(z) - torch.exp(-z)) / (torch.exp(z) + torch.exp(-z))
#plt.plot(tanh(tensor_A))
#plt.show()

# Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
# plt.show()

import torch
x = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long

# Create train and test splits
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=RANDOM_SEED)
acc_fn = Accuracy(task="multiclass", num_classes=3).to(device)

class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=10)
        self.linear2 = nn.Linear(in_features=10, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))

model_1 = SpiralModel().to(device)

x_train,y_train = x_train.to(device),y_train.to(device)
x_test,y_test = x_test.to(device),y_test.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(),lr=0.02)
epochs = 1000
for epoch in range(epochs):
    model_1.train()
    y_logits = model_1(x_train)
    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
    loss = loss_fn(y_logits,y_train)
    acc = acc_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(x_test)
        test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits,y_test)
        test_acc = acc_fn(y_test,test_pred)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_1,x_train,y_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_1,x_test,y_test)
# plt.show()