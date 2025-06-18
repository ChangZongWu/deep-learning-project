import pandas as pd
import torch
from jinja2.optimizer import optimize
from torch import nn
import torchvision
from torch.ao.quantization import compare_results
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from helper_functions import accuracy_fn

# torchvision - base domain library for pytorch computer vision
# torchvision.datasets - get datasets and data loading function for computer vision here
# torchvision.models - get pretrained computer vision models that you can leverage for ur own problems
# torchvision.transforms - functions for manipulating your vision data(image) to be suitable for use with an ML model
# torch.utils.data.Dataset
# torch.utils.data.Dataloader

# get dataset
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)
test_data = datasets.FashionMNIST(
    root="data",
    train= False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)
print(len(train_data))
print(len(test_data))
image, label = train_data[0]
print(image)
print(label)
class_names = train_data.classes
print(class_names)
class_to_idx = train_data.class_to_idx
print(class_to_idx)
print(f"Image shape: {image.shape} -> [Color_channels, height, width]")
print(label)

#visualizing data
# plt.imshow(image.squeeze())
# plt.title("label")
# plt.show()
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

#plot more images
# torch.manual_seed(42)
# fig = plt.figure(figsize=(9,9))
# rows, cols = 4,4
# for i in range(1,rows*cols+1):
#     random_idx = torch.randint(0,len(train_data),size=[1]).item()
#     img , label = train_data[random_idx]
#     fig.add_subplot(rows,cols,i)
#     plt.imshow(img.squeeze(),cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
# plt.show()

from torch.utils.data import DataLoader
# set up batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=False)

print(len(train_dataloader))
print(len(test_dataloader))
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_labels_batch.shape,train_features_batch.shape)

torch.manual_seed(42)
random_idx = torch.randint(0,len(train_features_batch),size=[1]).item()
img, label = train_features_batch[random_idx],train_labels_batch[random_idx]
# plt.imshow(img.squeeze(),cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")

# baseline model
# create a flatten layer
flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)
print(f"Shape before flattening:{x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape}-> [color_channels, height*width]")

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model_0 = FashionMNISTModelV0(input_shape=784,hidden_units=10 ,output_shape= len(class_names)).to("cpu")

dummy_x = torch.rand([1,1,28,28])
print(model_0(dummy_x))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)
from helper_functions import accuracy_fn

# create a func to time our experiments

from timeit import default_timer as timer
def print_train_time(start: float,end: float,device: torch.device = None):
    total_time = end - start
    print(f"Train Time on {device}: {total_time:.3f} seconds")
    return total_time

from tqdm.auto import tqdm
torch.manual_seed(42)
train_timer_start_on_cpu = timer()
epochs = 3
for epoch in range(epochs):
    train_loss = 0
    for batch, (x,y) in enumerate(train_dataloader):
        model_0.train()
        y_pred = model_0(x)
        loss = loss_fn(y_pred,y)
        train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if batch % 400 ==0:
            #print(f"Looked at {batch *len(x)}/{len(train_dataloader.dataset)} samples.")
    train_loss /= len(train_dataloader)
    test_loss, test_acc = 0,0
    model_0.eval()
    with torch.inference_mode():
        for x_test,y_test in test_dataloader:
            test_pred = model_0(x_test)
            test_loss += loss_fn(test_pred,y_test)
            test_acc += accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    #print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc:{test_acc:.4f}")
train_timer_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_timer_start_on_cpu,end=train_timer_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))

# make predictions and get result
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss, acc = 0,0
    model_0.eval()
    with torch.inference_mode():
        for x,y in data_loader:
            y_pred = model(x)
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true= y,y_pred = y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model name":model.__class__.__name__,
            "model loss": loss.item(),
            "model acc":acc}

model_0_result = eval_model(model=model_0,data_loader=test_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn)
print(model_0_result)

#set up device
device = "cuda"if torch.cuda.is_available() else "cpu"

# 6 model 1 : build better model with non-linearity
class FashionMNISTModelV1(nn.Module):
    def __init__(self,input_shape: int,hidden_units:int,output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_shape),
            nn.ReLU()
        )
    def forward(self,x: torch.Tensor):
        return self.layer_stack(x)

torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784,hidden_units=10, output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.1)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.train()
    # add a loop to loop through the training batches
    for batch, (x,y) in enumerate(data_loader):
        # put data on target device
        x,y = x.to(device),y.to(device)
        # 1 forward pass
        y_pred = model(x)
        #　2 Calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred,y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true = y,y_pred = y_pred.argmax(dim=1))# go from logits to prediction labels
        # 3 optimizer zero grad
        optimizer.zero_grad()
        # 4 loss backward
        loss.backward()
        # 5 optimizer step (update the model's parameters once per batch)
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc:{train_acc:.2f}%\n")

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    test_loss,test_acc = 0,0
    # put model in eval mode
    model.eval()
    # turn on inference mode context manager
    with torch.inference_mode():
        for x,y in data_loader:
            # send the data to the target device
            x,y = x.to(device),y.to(device)
            # 1. forward pass (outputs raw logits)
            test_pred = model(x)
            # 2 Calculate the loss/acc
            test_loss += loss_fn(test_pred,y)
            test_acc += accuracy_fn(y_true= y,y_pred=test_pred.argmax(dim=1))
        # adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc:{test_acc:.2f}%\n")

torch.manual_seed(42)
    # measure time
train_timer_start_on_gpu = timer()
epochs = 3
#for epoch in tqdm(range(epochs)):
for epoch in range(epochs):
    print(f"Epoch:{epoch}\n--------")
    train_step(model = model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    train_timer_end_on_gpu = timer()
    total_train_time_model_1 = print_train_time(start=train_timer_start_on_gpu,
                                                end=train_timer_end_on_gpu,
                                                device=device)

torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss, acc = 0,0
    model_0.eval()
    with torch.inference_mode():
        for x,y in data_loader:
            x,y = x.to(device),y.to(device)
            y_pred = model(x)
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true= y,y_pred = y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model name":model.__class__.__name__,
            "model loss": loss.item(),
            "model acc":acc}

model_1_result = eval_model(model= model_1,
                            data_loader=test_dataloader,
                            loss_fn=loss_fn,
                            accuracy_fn=accuracy_fn)
# print(model_1_result)
# print(model_0_result)

# model 2  Convolutional Neural Network (CNN)

class FashionMNISTModelV2(nn.Module):
    def __init__(self,input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.con_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.con_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )
    def forward(self,x):
        x = self.con_block_1(x)
        #print(x.shape)
        x = self.con_block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,output_shape= len(class_names), hidden_units= 10).to(device)
rand_image_tensor = torch.randn(size=(1,28,28))
model_2(rand_image_tensor.unsqueeze(0).to(device))
# 7.1 step through nn.Conv2d

torch.manual_seed(42)
images = torch.randn(size=(32,3,64,64))
test_image = images[0]
# print(f"Image batch shape: {images.shape}")
# print(f"Single image shape: {test_image.shape}")
# print(f"test image:\n {test_image}")

# single conv2d layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=10,kernel_size=3,stride=1,padding=0)
# pass the data
conv_output = conv_layer(test_image)
#conv_output = conv_layer(test_image.unsqueeze(0))
print(conv_output.shape)

# 7.2 stepping through nn.Maxpool2d
max_pool_layer = nn.MaxPool2d(kernel_size=2)
test_image_through_conv = conv_layer(test_image)
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
torch.manual_seed(42)
random_tensor = torch.randn(size=(1,1,2,2))
max_pool_layer = nn.MaxPool2d(kernel_size=2)
max_pool_tensor = max_pool_layer(random_tensor)

# 7.3 loss optim for model2
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.1)
# 7.3 train test loop for model2
torch.manual_seed(42)
torch.cuda.manual_seed(42)
train_timer_start_model_2 = timer()
epochs =3
for epoch in range(epochs):
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    train_timer_end_model_2 = timer()
    total_train_time_model_2 = print_train_time(start=train_timer_start_model_2,
                                                end=train_timer_end_model_2,
                                                device=device)
model_2_result = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
print(model_2_result)
print(model_1_result)
print(model_0_result)

compare_results = pd.DataFrame([model_0_result,
                                model_1_result,
                                model_2_result])

# add training time to results comparison
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]
print(compare_results)

# plot the results
# compare_results.set_index("model name")["model acc"].plot(kind="barh")
# plt.xlabel("accuracy (%)")
# plt.ylabel("model")
# plt.show()

# make and evaluate random predictions with best model
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device)  # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(),
                                      dim=0)  # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

import  random
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(label)

# plt.imshow(test_samples[0].squeeze(), cmap="gray")
# plt.show()
# make predictions
pred_probs = make_predictions(model=model_2,
                              data=test_samples)
# convert prediction prob to labels
pred_classes = pred_probs.argmax(dim=1)
# plot predictions
# plt.figure(figsize=(9,9))
# nrow = 3
# ncol = 3
# for i, sample in enumerate(test_samples):
#     plt.subplot(nrow,ncol, i+1)
#     plt.imshow(sample.squeeze(), cmap = "gray")
#     pred_label = class_names[pred_classes[i]]
#     truth_label = class_names[test_labels[i]]
#     title_text = f"Pred: {pred_label} | Truth: {truth_label}"
#     if pred_label == truth_label:
#         plt.title(title_text,fontsize=10,c="g")
#     else:
#         plt.title(title_text, fontsize=10, c="r")
#
# plt.show()

# 10 making a confusion matrix
# 1. make prediction with trained model


import mlxtend
print(mlxtend.__version__)
y_preds = []
model_2.eval()
with torch.inference_mode():
    for x,y in tqdm(test_dataloader, desc = "Making pred...."):
        x,y= x.to(device) ,y.to(device)
        y_logit = model_2(x)
        y_pred = torch.softmax(y_logit.squeeze(),dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())
#print(y_preds)
y_pred_tensor = torch.cat(y_preds)
import  torchmetrics
from  torchmetrics import  ConfusionMatrix
from mlxtend.plotting import  plot_confusion_matrix
confmat =ConfusionMatrix(num_classes=len(class_names),task="multiclass")
confmat_tensor = confmat(preds = y_pred_tensor,
                         target = test_data.targets)
fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                class_names=class_names,
                                figsize=(10,7))
# 11. save and load best performing model
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)

# create a new instance
torch.manual_seed(42)
loaded_model_2= FashionMNISTModelV2(input_shape=1,
                                    hidden_units=10,
                                    output_shape=len(class_names))
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_2.to(device)

loaded_model_2_result = eval_model(model=loaded_model_2,data_loader=test_dataloader,
                                   loss_fn=loss_fn,
                                   accuracy_fn=accuracy_fn)
print(model_2_result)
print(loaded_model_2_result)

# check if model results are close to each other
print(torch.isclose(torch.tensor(model_2_result["model loss"]),
              torch.tensor(loaded_model_2_result["model loss"]),
              atol=1e-02))

------------------------------------exercise-----------------------------------------

# Import torch
import torch
from jinja2.optimizer import optimize
from mpmath.identification import transforms

from helper_functions import download_data, accuracy_fn

# TODO: Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. What are 3 areas in industry where computer vision is currently being used?
# 1.Self-driving cars 2.Healthcare imaging 3.Security

# 2. Search "what is overfitting in machine learning" and write down a sentence about what you find.
# creating a model that matches (memorizes) the training set so closely
# that the model fails to make correct predictions on new data

# 3. Search "ways to prevent overfitting in machine learning",
# write down 3 of the things you find and a sentence about each.
# 1.Early stopping 2.Data augmentation 3.Increase training data

# 4. Spend 20-minutes reading and clicking through the CNN Explainer website.

# 5. Load the torchvision.datasets.MNIST() train and test datasets.

from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_data = datasets.MNIST(
    root="data", # where to download data to?
    train= True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

test_data = datasets.MNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)
print(len(train_data))
print(len(test_data))
img = train_data[0][0]
label = train_data[0][1]
#print(f"Image:\n {img}")
#print(f"Label:\n {label}")
# Check out the shapes of our data
print(f"Image shape: {img.shape} -> [color_channels, height, width] (CHW)")
print(f"Label: {label} -> no shape, due to being integer")
class_names = train_data.classes
print(class_names)
# 6. Visualize at least 5 different samples of the MNIST training dataset.

# for i in range(5):
#     img = train_data[i][0]
#     img_squeeze = img.squeeze()
#     label = train_data[i][1]
#     plt.figure(figsize=(3,3))
#     plt.imshow(img_squeeze,cmap="gray")
#     plt.title(label)
#     plt.axis(False)
# plt.show()

# 7. Turn the MNIST train and test datasets into dataloaders using torch.
# utils.data.DataLoader, set the batch_size=32.

from torch.utils.data import DataLoader
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
                              batch_size=BATCH_SIZE, # how many samples per batch?
                              shuffle=True) # shuffle data every epoch?
test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)# don't necessarily have to shuffle the testing data
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape)
print(train_labels_batch.shape)

# 8. Recreate model_2 used in notebook 03
# (the same model from the CNN Explainer website, also known as TinyVGG)
# capable of fitting on the MNIST dataset.

class MNISTModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"Output shape of conv block 1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape of conv block 2: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape of classifier: {x.shape}")
        return x

model = MNISTModel(input_shape=1,output_shape=10,hidden_units=10).to(device)

# 9. Train the model you built in exercise 8. for 5 epochs on CPU and GPU and see how long it takes on each.

from tqdm.auto import tqdm
from tqdm.auto import tqdm

# Train on CPU
model_cpu = MNISTModel(input_shape=1,
                        hidden_units=10,
                        output_shape=10).to("cpu")

# Create a loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_cpu.parameters(), lr=0.1)

### Training loop
epochs = 5
for epoch in tqdm(range(epochs)):
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model_cpu.train()

        # Put data on CPU
        X, y = X.to("cpu"), y.to("cpu")

        # Forward pass
        y_pred = model_cpu(X)

        # Loss calculation
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Step the optimizer
        optimizer.step()

    # Adjust train loss for number of batches
    train_loss /= len(train_dataloader)

    ### Testing loop
    test_loss_total = 0

    # Put model in eval mode
    model_cpu.eval()

    # Turn on inference mode
    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(test_dataloader):
            # Make sure test data on CPU
            X_test, y_test = X_test.to("cpu"), y_test.to("cpu")
            test_pred = model_cpu(X_test)
            test_loss = loss_fn(test_pred, y_test)

            test_loss_total += test_loss

        test_loss_total /= len(test_dataloader)

    # Print out what's happening
    print(f"Epoch: {epoch} | Loss: {train_loss:.3f} | Test loss: {test_loss_total:.3f}")

from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Train on GPU
model_gpu = MNISTModel(input_shape=1,
                        hidden_units=10,
                        output_shape=10).to(device)

# Create a loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_gpu.parameters(), lr=0.1)

# Training loop
epochs = 5
for epoch in tqdm(range(epochs)):
    train_loss = 0
    model_gpu.train()
    for batch, (X, y) in enumerate(train_dataloader):
        # Put data on target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model_gpu(X)

        # Loss calculation
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Step the optimizer
        optimizer.step()

    # Adjust train loss to number of batches
    train_loss /= len(train_dataloader)

    ### Testing loop
    test_loss_total = 0
    # Put model in eval mode and turn on inference mode
    model_gpu.eval()
    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(test_dataloader):
            # Make sure test data on target device
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_pred = model_gpu(X_test)
            test_loss = loss_fn(test_pred, y_test)

            test_loss_total += test_loss

        # Adjust test loss total for number of batches
        test_loss_total /= len(test_dataloader)

    # Print out what's happening
    print(f"Epoch: {epoch} | Loss: {train_loss:.3f} | Test loss: {test_loss_total:.3f}")

# 10. Make predictions using your trained model and visualize at least 5 of them comparing
# the prediciton to the target label.

num_to_plot = 5
for i in range(num_to_plot):
  # Get image and labels from the test data
  img = test_data[i][0]
  label = test_data[i][1]

  # Make prediction on image
  model_pred_logits = model_gpu(img.unsqueeze(dim=0).to(device))
  model_pred_probs = torch.softmax(model_pred_logits, dim=1)
  model_pred_label = torch.argmax(model_pred_probs, dim=1)

  # Plot the image and prediction
#   plt.figure()
#   plt.imshow(img.squeeze(), cmap="gray")
#   plt.title(f"Truth: {label} | Pred: {model_pred_label.cpu().item()}")
#   plt.axis(False)
# plt.show()

# 11. Plot a confusion matrix comparing your model's predictions to the truth labels.
import torchmetrics, mlxtend
model_gpu.eval()
y_preds = []
with torch.inference_mode():
    for batch, (X,y) in enumerate(test_dataloader):
        X,y = X.to(device),y.to(device)
        y_pred_logits = model_gpu(X)
        y_pred_labels = torch.argmax(torch.softmax(y_pred_logits,dim=1),dim=1)
        y_preds.append(y_pred_labels)
    y_preds=torch.cat(y_preds).cpu()
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
# confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
# confmat_tensor = confmat(preds= y_preds,
#                          target=test_data.targets)
# fix, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
#                                 class_names=class_names,
#                                 figsize=(10,7))
# plt.show()
# 12. Create a random tensor of shape [1, 3, 64, 64] and
# pass it through a nn.Conv2d() layer with various hyperparameter settings
# (these can be any settings you choose),
# what do you notice if the kernel_size parameter goes up and down?

random_tensor = torch.rand([1, 3, 64, 64])
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=64,
                       kernel_size=2,
                       stride=2,
                       padding=1)
print(f"Random tensor original shape: {random_tensor.shape}")
random_tensor_through_conv_layer = conv_layer(random_tensor)
print(f"Random tensor through conv layer shape: {random_tensor_through_conv_layer.shape}")
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=64,
                       kernel_size=3,
                       stride=2,
                       padding=1)
print(f"Random tensor original shape: {random_tensor.shape}")
random_tensor_through_conv_layer = conv_layer(random_tensor)
print(f"Random tensor through conv layer shape: {random_tensor_through_conv_layer.shape}")
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=64,
                       kernel_size=4,
                       stride=2,
                       padding=1)
print(f"Random tensor original shape: {random_tensor.shape}")
random_tensor_through_conv_layer = conv_layer(random_tensor)
print(f"Random tensor through conv layer shape: {random_tensor_through_conv_layer.shape}")

# 13. Use a model similar to the trained model_2 from notebook 03 to
# make predictions on the test torchvision.datasets.FashionMNIST dataset.
from torchvision import transforms
fashion_mnist_train = datasets.FashionMNIST(root="data",
                                            download=True,
                                            train=True,
                                            transform=transforms.ToTensor())
fashion_mnist_test = datasets.FashionMNIST(root="data",
                                           train=False,
                                           download=True,
                                           transform=transforms.ToTensor())
fashion_mnist_class_names = fashion_mnist_train.classes
fashion_mnist_train_dataloader = DataLoader(fashion_mnist_train,
                                            batch_size=32,shuffle=True)
fashion_mnist_test_dataloader = DataLoader(fashion_mnist_test,
                                           batch_size=32,shuffle=False)
model_2 = MNISTModel(input_shape=1,
                     output_shape=10,
                     hidden_units=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_2.parameters(),lr=0.01)
from torchmetrics import Accuracy
acc_fn = Accuracy(task = 'multiclass', num_classes=len(fashion_mnist_class_names)).to(device)
epochs = 5
for epoch in tqdm(range(epochs)):
    train_loss, test_loss_total = 0, 0
    train_acc, test_acc = 0, 0

    ### Training
    model_2.train()
    for batch, (X_train, y_train) in enumerate(fashion_mnist_train_dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass and loss
        y_pred = model_2(X_train)
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        train_acc += acc_fn(y_pred, y_train)

        # Backprop and gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Adjust the loss/acc (find the loss/acc per epoch)
    train_loss /= len(fashion_mnist_train_dataloader)
    train_acc /= len(fashion_mnist_train_dataloader)

    ### Testing
    model_2.eval()
    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(fashion_mnist_test_dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Forward pass and loss
            y_pred_test = model_2(X_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_loss_total += test_loss

            test_acc += acc_fn(y_pred_test, y_test)

        # Adjust the loss/acc (find the loss/acc per epoch)
        test_loss /= len(fashion_mnist_test_dataloader)
        test_acc /= len(fashion_mnist_test_dataloader)

    # Print out what's happening
    print(
        f"Epoch: {epoch} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.2f} | Test loss: {test_loss_total:.3f} | Test acc: {test_acc:.2f}")

test_preds = []
model_2.eval()
with torch.inference_mode():
    for X_test, y_test in fashion_mnist_test_dataloader:
        y_logits = model_2(X_test.to(device))
        y_pred_probs = torch.softmax(y_logits,dim=1)
        y_pred_labels = torch.argmax(y_pred_probs,dim=1)
        test_preds.append(y_pred_labels)
test_preds = torch.cat(test_preds).cpu()
import numpy as np
wrong_pred_indexes = np.where(test_preds != fashion_mnist_test.targets)[0]
import random
random_selection = random.sample(list(wrong_pred_indexes), k=9)

plt.figure(figsize=(10, 10))
for i, idx in enumerate(random_selection):
  # Get true and pred labels
  true_label = fashion_mnist_class_names[fashion_mnist_test[idx][1]]
  pred_label = fashion_mnist_class_names[test_preds[idx]]

  # Plot the wrong prediction with its original label
  plt.subplot(3, 3, i+1)
  plt.imshow(fashion_mnist_test[idx][0].squeeze(), cmap="gray")
  plt.title(f"True: {true_label} | Pred: {pred_label}", c="r")
  plt.axis(False)
plt.show()