import torch
from  torch import nn # nn contains all of pytorch's building block for neural networks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import device
from pathlib import Path

# Data (preparing and loading)
# Data can be almost anything in ML

# Excel spreadsheet
# Image
# Videos
# Audio
# DNA
# Text
# ML -
# 1.get data into a numerical representation
# 2 Build a model to learn patterns in that numerical representation

# create *known* parameters
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
x = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * x + bias

print(x[:10])
print(y[:10])
print(len(x))
print(len(y))

# splitting data into training and test sets
# create a training and test set with our data

train_split = int(0.8 * len(x))
print(train_split)
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test =  x[train_split:], y[train_split:]

print(len(x_train))
print(x_train)
print(len(y_train))
print(y_train)
print(len(x_test))
print(x_test)
print(len(y_test))
print(y_test)


def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14});
    plt.show()

#plot_predictions()

# build model
# create linear regression model class
class LinearRegressionModel(nn.Module): # almost everything in pytorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))
        #Forward method to define the computation in the model
    def forward(self, x:torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weights * x + self.bias

# pytorch model building essentials
# torch.nn - contains all of the buildings for computational graphs
# (a neural network can be considered a computational graph)
# torch.nn.Parameter - what parameters should our model try and learn, often a pytorch layer from torch.nn will set these for us
# torch.nn.Module - the base class for all neural network modules, if you subclass it, you should overwrite forward()
# torch.optim - this where the optimizers in pytorch live, they will help with gradient descent
# def forward() = All nn.Module subclasses require you to overwrite forward(), this method defines what happens in the forward computation

torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))

# list named parameters
print(model_0.state_dict())

#make prediction using torch.inference_mode()

with torch.inference_mode():
    y_preds = model_0(x_test)
# can also use
# with torch.no_grad():
#     y_preds = model_0(x_test)

print(y_preds)
#plot_predictions(y_preds)

# train model is to let model to move from some "unknown" parameters to some "known" parameters
# to measure how poor your models predictions are then we can use loss function

#set up loss function
loss_fn = nn.L1Loss()

#set up optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)

# build training loop and testing loop


torch.manual_seed(42)
# an epochs is one loop through the data
epochs = 200
# track different values
epoch_count = []
loss_val = []
test_loss_val = []

# 0 loop through data
for epoch in range(epochs):
    #set model to training mode
    model_0.train()
    # 1 forward pass (to maek prediction)
    y_preds = model_0(x_train)
    # 2 calculate the loss
    loss = loss_fn(y_preds,y_train)
    #print(f"Loss: {loss}")
    # 3 optimizer zero grad
    optimizer.zero_grad()
    # 4 loss backward (backpropagation)
    loss.backward()
    # 5 optimizer step (gradient descent)
    optimizer.step()
    # test
    model_0.eval()
    with torch.inference_mode():
        # 1 forward pass
        test_pred = model_0(x_test)
        # 2 calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch %10 == 0 :
        epoch_count.append(epoch)
        loss_val.append(loss)
        test_loss_val.append(test_loss)
        #print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        #print(model_0.state_dict())
with torch.inference_mode():
    y_preds_new = model_0(x_test)

#plot_predictions(predictions=y_preds_new)
def plot_training_and_loss(epoch = epoch_count, loss = loss_val, test_loss = test_loss_val):
    plt.plot(epoch, np.array(torch.tensor(loss).numpy()), label="Train loss")
    plt.plot(epoch, test_loss, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

#plot_training_and_loss()


# save a model in pytorch
# 1. torch.save() - allows you save a pytorch object in python's pickle format
# 2. torch.load() - allows you load a saved pytorch object
# 3. torch.nn.Module.load_state_dict() - this allows to load a model's saved state dictionary

# create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# create model save path
# it will also work if you use .pt file
MODEL_NAME ="01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the model state dict
torch.save(obj=model_0.state_dict(),f= MODEL_SAVE_PATH)

# loading pytorch model
# to load in a saved a state_dict we have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel()

# load the saved state_dict of model_0 (this will update the new instance with updated parameter)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only= True))

print(loaded_model_0.state_dict())

# make some predictions with our loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(x_test)

print(loaded_model_preds)

# make some models preds
model_0.eval()
with torch.inference_mode():
    y_preds= model_0(x_test)

print(y_preds==loaded_model_preds)

# go through the steps above

# import pytorch and matplotlib
import torch
from torch import nn
import matplotlib.pyplot as plt
# create  a device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# 6.1 Data
# create some data using linear regression formula of y = weight * x + bias
weight = 0.7
bias = 0.3

# create range of val
start = 0
end = 1
step = 0.02

# create x and y
x = torch.arange(start,end, step).unsqueeze(dim=1)
y= weight * x + bias

# split data
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split],y[:train_split]
x_test, y_test = x[train_split:],y[train_split:]

# plot the data
#plot_predictions(x_train,y_train,x_test,y_test)

# 6.2 build a pytorch linear model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # use nn.Linear () for creating the model parameters / also called: linear transform
        self.linear_layer = nn.Linear(in_features=1,out_features=1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# set the manual seed
torch.manual_seed(42)
model_1 = LinearModel()
print(model_1)
print(model_1.state_dict())

#set the model to use the target device
model_1.to(device)
print(next(model_1.parameters()).device)

# 6.3 training
# loss function
# optimizer
# training loop
# testing loop

# set up loss function
loss_fn = nn.L1Loss()

# set up optimizer
optimizer = torch.optim.SGD(params= model_1.parameters(),lr=0.01)

# training loop
torch.manual_seed(42)

epochs = 200

# put data on the target device (device agnostic code for data)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_1.train()
    # 1 forward pass
    y_preds = model_1(x_train)

    # 2 calculate the loss
    loss = loss_fn(y_preds, y_train)

    # 3 optimizer zero grad
    optimizer.zero_grad()

    # 4. perform backpropagation
    loss.backward()

    # 5. optimizer step
    optimizer.step()

    # testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(x_test)

        test_loss = loss_fn(test_pred, y_test)

    # print out
    if epoch%10 ==0:
        print(f"Epoch:{epoch} | Loss:{loss} | Test loss:{test_loss}")

print(model_1.state_dict())

# 6.4 make and evaluate predictions

# turn model into evaluation mode
model_1.eval()

# make predictions on the test data
with torch.inference_mode():
    y_preds=model_1(x_test)

print(y_preds)

# check out our model predictions visually
#plot_predictions(predictions=y_preds.cpu())

# 6.5 saving and loading model
from pathlib import Path

# 1. create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

# 2. create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. save the model state dict
torch.save(obj=model_1.state_dict(), f= MODEL_SAVE_PATH)

# load a pytorch
# create a new instance of linear regression model
loaded_model_1 = LinearModel()
# load the saved model_1 state_dict
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH,weights_only=True))

# put the loaded model to device
loaded_model_1.to(device)

print(loaded_model_1.state_dict())

# evaluat the model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(x_test)

print(y_preds == loaded_model_1_preds)

# exercises
# Import necessary libraries
import torch
from torch import nn
import matplotlib.pyplot as plt
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
#Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
#Split the data into 80% training, 20% testing.
#Plot the training and testing data so it becomes visual.
# Create the data parameters
weight = 0.3
bias = 0.9
start = 0
end = 1
step = 0.01
# Make X and y using linear regression feature
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias
print(f"Number of X samples: {len(x)}")
print(f"Number of y samples: {len(y)}")
print(f"First 10 X & y samples:\nX: {x[:10]}\ny: {y[:10]}")
# Split the data into training and testing
train_split = int(0.8 * len(x))
x_train = x[:train_split]
y_train = y[:train_split]
x_test = x[train_split:]
y_test = y[train_split:]
# Plot the training and testing data
def plot_predictions(train_data = x_train,
                 train_labels = y_train,
                 test_data = x_test,
                 test_labels = y_test,
                 predictions = None):
  plt.figure(figsize = (10,7))
  plt.scatter(train_data,train_labels,c = 'b',s = 4,label = "Training data")
  plt.scatter(test_data,test_labels,c = 'g',s = 4,label = "Test data")

  if predictions is not None:
    plt.scatter(test_data,predictions,c = 'r',s = 4,label = "Predictions")
  plt.legend(prop = {"size" : 14})
  plt.show()

# Create PyTorch linear regression model by subclassing nn.Module
class LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
    def forward(self, x):
        return self.weights * x + self.bias

# Instantiate the model and put it to the target device
torch.manual_seed(42)
model_2 = LinearRegressionV2()
model_2.to(device)

# Create the loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.01)

# Training loop

torch.manual_seed(42)
# Train model for 300 epochs
epochs = 300

# Send data to target device
x_test=x_test.to(device)
y_test=y_test.to(device)
x_train=x_train.to(device)
y_train=y_train.to(device)

for epoch in range(epochs):
    ### Training

    # Put model in train mode
    model_2.train()
    # 1. Forward pass
    y_pred = model_2(x_train)
    # 2. Calculate loss
    loss = loss_fn(y_pred,y_train)
    # 3. Zero gradients
    optimizer.zero_grad()
    # 4. Backpropagation
    loss.backward()
    # 5. Step the optimizer
    optimizer.step()
    ### Perform testing every 20 epochs
    if epoch % 20 == 0:
        # Put model in evaluation mode and setup inference context
        model_2.eval()
        # 1. Forward pass
        test_preds = model_2(x_test)
        # 2. Calculate test loss
        test_loss = loss_fn(test_preds,y_test)
        # Print out what's happening
        print(f"Epoch: {epoch} | Train loss: {loss:.3f} | Test loss: {test_loss:.3f}")

# Make predictions with the model
model_2.eval()
with torch.inference_mode():
    y_preds = model_2(x_test)
# Plot the predictions (these may need to be on a specific device)
#plot_predictions(predictions=y_preds.cpu())

from pathlib import Path
# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)
# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_2.pth"
# 3. Save the model state dict
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_2.state_dict(),f=MODEL_SAVE_PATH)
# Create new instance of model and load saved state dict (make sure to put it on the target device)
loaded_model_2 = LinearRegressionV2()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH,weights_only=True))
loaded_model_2.to(device)
# Make predictions with loaded model and compare them to the previous
loaded_model_2.eval()
with torch.inference_mode():
    loaded_model_2_preds = loaded_model_2(x_test)
print(y_preds==loaded_model_2_preds)