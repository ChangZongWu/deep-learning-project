import torch
import torchvision
from torch import nn
from torch.cuda import device


device = "cuda" if torch.cuda.is_available() else "cpu"
# 1. get the dataset

import requests
import zipfile
from pathlib import Path

#set the path to the dataset
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"Dataset already exists at {image_path}")
else:
    print(f"Dataset not found. Downloading...")
    image_path.mkdir(parents=True,exist_ok=True)

# download the dataset
with open(data_path/"pizza_steak_sushi.zip", "wb") as f:
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip"
    response = requests.get(url)
    f.write(response.content)

# unzip the dataset
with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip", "r") as zip_ref:
    zip_ref.extractall(image_path)

# 2 data preparation and data exploration
import os
def walk_through_dir(dir_path: str):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# 2.1 visualize the images
import random
from PIL import Image

# random.seed(42)
# 1.get all the image paths
image_path_list = list(train_dir.glob("*/*.jpg"))
# 2.get a random image path
random_image_path = random.choice(image_path_list)
# 3. get the image class from the path name
image_class = random_image_path.parent.stem
# 4. open the image
image = Image.open(random_image_path)
# 5. print the metadata
print(f"Image size: {image.size}")
print(f"Image class: {image_class}")
print(f"Image height: {image.height}")
print(f"Image width: {image.width}")

import numpy as np
import matplotlib.pyplot as plt

# Turn the image into a numpy array and display it
img_as_array = np.asarray(image)
# plot the image
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]" )
# plt.axis(False)
# plt.show()

# 3. Transform the data
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

# 3.1 Transform the data
data_transform = transforms.Compose([
    transforms.Resize((64,64)),  # Resize the image to 64x64 pixels
    transforms.RandomHorizontalFlip(p=0.5) ,  # Randomly flip the image horizontally with a probability of 0.5
    transforms.ToTensor(),           # Convert the image to a tensor
])
print(data_transform(image).shape)  # Check the shape of the transformed image tensor
def plot_transformed_image(image_paths: list, transform, n=3, seed=None):
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths,k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize:{ f.size}")
            ax[0].axis("off")
            transformed_image = transform(f).permute(1, 2, 0)  # Change the order of dimensions from [C, H, W] to [H, W, C] for displaying
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")  # shape is [C, H, W]
            ax[1].axis("off")
            fig.suptitle(f"Transformed Image from {image_path.name}", fontsize=16)

# plot_transformed_image(image_paths=image_path_list,
#                        transform=data_transform,
#                        n=3,seed=42)
# plt.show()

# 4. option1. load the image dataset using torchvision.datasets.ImageFolder
train_data = datasets.ImageFolder(root=train_dir, transform=data_transform,target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform, target_transform=None)

# print(train_data, test_data)
# get class names as a list
class_names = train_data.classes
# get class names as a dictionary
class_dict = train_data.class_to_idx
img, label = train_data[0][0], train_data[0][1]
print(f"Image: {img}")
print(f"Image shape: {img.shape}")  # should be [C, H, W]
print(f"image label: {label}")  # label is an integer corresponding to the class index
print(f"Image class name: {class_names[label]}")  # get the class name from the label
img_permute = img.permute(1, 2, 0)  # Change the order of dimensions from [C, H, W] to [H, W, C] for displaying
print(f"Image shape before permute: {img.shape}")  # should be [C, H, W]
print(f"Image shape after permute: {img_permute.shape}")  # should be [H, W, C]
# plot the image
# plt.figure(figsize=(6, 6))
# plt.imshow( img_permute)
# plt.axis("off")
# plt.title(class_names[label])
# plt.show()

# 4.1 turn loaded image dataset into dataloaders
import os
print(os.cpu_count())
from torch.utils.data import DataLoader
BATCH_SIZE = 1
train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=1)
test_loader = DataLoader(dataset=test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=1)
# img,label = next(iter(train_loader))  # get a batch of data
# print(f"Batch of images shape: {img.shape}")  # should be [B, C, H, W]
# print(f"Batch of labels shape: {label.shape}")  # should be [B]

# 5 option2. load the image dataset using custom
import os
import  pathlib
from typing import Tuple,List, Dict
# 5.1 create a helper function to get class name
target_directtory = train_dir
class_name_found = sorted([entry.name for entry in list(os.scandir(target_directtory))])
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory)if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f" No classes found in directory: {directory}. Make sure the directory contains subdirectories for each class.")
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    return classes, class_to_idx

# 5.2 create a custom dataset to replicate ImageFolder
from torch.utils.data import Dataset
# 1. sunclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))  # Get all jpg images in subdirectories
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    def load_image(self,index: int) -> Image.Image:
        img_path = self.paths[index]
        return Image.open(img_path)
    def __len__(self) -> int:
        return len(self.paths)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
# create a transform
train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]
)
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# test out the ImageFolderCustom
train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

# 5.3 create a func to display random images
def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str]=None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    if n>10:
        n=10
        display_shape = False
    if seed:
        random.seed(seed)
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16,8))
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        targ_image_adjust = targ_image.permute(1, 2, 0)
        plt.subplot(1,n,i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nShape: {targ_image.shape}"
        plt.title(title)

# display_random_images(train_data,
#                       n=5,
#                       classes=class_names,
#                       seed=None)
#plt.show()
# display_random_images(test_data_custom,
#                       n=20,
#                       classes=class_names,
#                       seed=42)
# plt.show()
# 5.4 turn custom loaded image into dataloaders

train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=1)
test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=1)
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()])
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()])
image_path_list = list(image_path.glob("*/*/*.jpg"))
# plot_transformed_image(image_paths=image_path_list,
#                        transform=train_transform,
#                        n=3,seed=None)
# plt.show()
# 7 model 0: TinyVGG without data augmentation
# 7.1 create transform and loading data for model 0

simple_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()])
from torchvision import datasets
train_data_simple = datasets.ImageFolder(root=train_dir,
                                         transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=simple_transform)

import os
from torch.utils.data import DataLoader
BATCH_SIZE = 32
# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0
train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

# 7.2 CREATE A TINYVGG MODEL
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape),
            nn.ReLU())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        #print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x

model_0 = TinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=len(class_names)).to(device)

# 7.3 try forward pass
# image_batch, label_batch = train_dataloader_simple.__iter__().__next__()
# print(image_batch.shape, label_batch.shape)
# model_0(image_batch.to(device))

# 7.4 use torch info to get idea of the shape of the model
import torchinfo
from torchinfo import summary
summary(model_0,input_size=[1,3,64,64])
from helper_functions import accuracy_fn
def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):
    model.train()
    total_loss, total_acc = 0,0
    for batch ,(X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        total_acc += (y_pred_class==y).sum().item()/len(y_pred)
    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    return total_loss, total_acc

def test_step(model:nn.Module,
              dataloader: DataLoader,
              loss_fn: nn.Module,
              device: torch.device = device):
    total_loss, total_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred,y)
            total_loss+=loss.item()
            y_pred_class = torch.argmax(y_pred, dim=1)
            total_acc += (y_pred_class==y).sum().item()/len(y_pred)
    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    return total_loss, total_acc
from tqdm.auto import tqdm
def train(model:nn.Module,
          optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: torch.device = device):
    result = {"train_loss": [],
              "train_acc": [],
              "test_loss": [],
              "test_acc": []}
    for epoch in range(epochs):
        train_loss, train_acc =train_step(model,train_dataloader,
                   loss_fn,optimizer,device)
        test_loss, test_acc =test_step(model,test_dataloader,
                  loss_fn,device)
        print(f"Epoch:{epoch}| train_loss:{train_loss:.4f}| train_acc:{train_acc:.4f}| test_loss:{test_loss:.4f}| test_acc:{test_acc:.4f}")
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        result["test_loss"].append(test_loss)
        result["test_acc"].append(test_acc)
    return result

# 7.7 train and evaluate the model
NUM_EPOCHS = 5
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(),lr=0.001)
from timeit import default_timer as timer
start_time = timer()
model_0_results = train(model=model_0,
                        optimizer=optimizer,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS,
                        device=device)
end_time = timer()
total_time = end_time-start_time
print(f"Total training time: {total_time:.3f} seconds")
# 7.8 plot the loss curves
import matplotlib.pyplot as plt
model_0_results.keys()
def plot_loss_curves(results: Dict[str,List[float]]):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    acc = results["train_acc"]
    test_acc = results["test_acc"]
    epochs = range(len(loss))
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label="train_loss")
    plt.plot(epochs,test_loss,label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs,acc,label="train_acc")
    plt.plot(epochs,test_acc,label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
# plot_loss_curves(model_0_results)

# create model with data augmentation
# create a training transform
train_transform_trivial = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()])
test_transforms_simple = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()])
# create a training and test data
train_data_trivial = datasets.ImageFolder(root=train_dir,
                                         transform=train_transform_trivial)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=test_transforms_simple)
# create a dataloader
train_dataloader_trivial = DataLoader(dataset=train_data_trivial,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)
model_1 = TinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=len(train_data_trivial.classes)).to(device)
start_time = timer()
model_1_results = train(model=model_1,
                        optimizer=optimizer,
                        train_dataloader=train_dataloader_trivial,
                        test_dataloader=test_dataloader_simple,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS,
                        device=device)
end_time = timer()
total_time = end_time-start_time
print(f"Total training time: {total_time:.3f} seconds")
# plot_loss_curves(model_1_results)
import pandas as pd
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)
# plt.figure(figsize=(15,10))
# epochs = range(len(model_0_df))
# plt.subplot(2,2,1)
# plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
# plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
# plt.title("Train Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.subplot(2,2,2)
# plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
# plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
# plt.title("Test Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.subplot(2,2,3)
# plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
# plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
# plt.title("Train Accuracy")
# plt.xlabel("Epochs")
# plt.legend()
# plt.subplot(2,2,4)
# plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
# plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
# plt.title("Test Accuracy")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

# making a prediction on a custom image
import requests
custom_image_path = data_path/ "04-pizza-dad.jpeg"
if not custom_image_path.is_file():
    print(f"Image not found. Downloading...")
    with open(custom_image_path, "wb") as f:
        url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg"
        response = requests.get(url)
        f.write(response.content)
else:
    print(f"Image already exists at {custom_image_path}")
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))
# plt.imshow(custom_image_uint8.permute(1,2,0))
# plt.show()
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)/255.0
model_1.eval()
custom_image_transform = transforms.Compose([
    transforms.Resize((64,64))
])
custom_image_transformed = custom_image_transform(custom_image)
with torch.inference_mode():
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(0).to(device))
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
custom_image_pred_labels = torch.argmax(custom_image_pred_probs, dim=1).cpu()
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform = None,
                        device = device):
    target_image = torchvision.io.read_image(image_path).type(torch.float32)/255.0
    if transform:
        target_image = transform(target_image)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(0).to(device)
        target_image_pred = model(target_image)

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    plt.imshow(target_image.squeeze().permute(1,2,0))
    if class_names:
        plt.title(f"Predicted: {class_names[target_image_pred_label.cpu()]} | probability: {target_image_pred_probs.max().cpu():.3f}")
    else:
        plt.title(f"Predicted: {target_image_pred_label.cpu()} | probability: {target_image_pred_probs.max().cpu():.3f}")
    plt.axis(False)
    plt.show()

pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform)





