import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader



train_data = datasets.FashionMNIST(
    root="data",
    train = True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
print(len(train_data), len(test_data))
image, label = train_data[0]
print(image.shape)
print(label)
print(len(train_data.data), len(train_data.targets))
print(len(test_data.data), len(test_data.targets))
class_names = train_data.classes
print(class_names)
print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.show()
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
# plt.show()
train_dataloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)
test_dataloader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False)
print(train_dataloader,test_dataloader)
print(len(train_data), len(test_data))
print(len(train_dataloader), len(test_dataloader))
device = "cuda" if torch.cuda.is_available() else "cpu"
from helper_functions import accuracy_fn
def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device:torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss+=loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred = y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}%")

def test_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               accuracy_fn,
               device:torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss+=loss
            test_acc += accuracy_fn(y_true=y,
                                    y_pred = y_pred.argmax(dim=1))
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")

class FashionMNISTModel(nn.Module):
    def __init__(self , input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.block_1 = nn.Sequential(
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
                      padding=1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
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
                      padding=1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classification_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classification_layer(x)
        return x
model = FashionMNISTModel(input_shape=1,
                            hidden_units=10,
                            output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                             lr=0.1)
epochs = 5
for epoch in range(epochs):
    train_step(model=model,
               dataloader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model,
              dataloader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)


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

import random
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs= make_predictions(model=model,
                             data=test_samples)
pred_classes = pred_probs.argmax(dim=1)
# Plot predictions
# plt.figure(figsize=(9, 9))
# nrows = 3
# ncols = 3
# for i, sample in enumerate(test_samples):
#     # Create a subplot
#     plt.subplot(nrows, ncols, i + 1)
#
#     # Plot the target image
#     plt.imshow(sample.squeeze(), cmap="gray")
#
#     # Find the prediction label (in text form, e.g. "Sandal")
#     pred_label = class_names[pred_classes[i]]
#
#     # Get the truth label (in text form, e.g. "T-shirt")
#     truth_label = class_names[test_labels[i]]
#
#     # Create the title text of the plot
#     title_text = f"Pred: {pred_label} | Truth: {truth_label}"
#
#     # Check for equality and change title colour accordingly
#     if pred_label == truth_label:
#         plt.title(title_text, fontsize=10, c="g")  # green text if correct
#     else:
#         plt.title(title_text, fontsize=10, c="r")  # red text if wrong
#     plt.axis(False)
# plt.show()

y_preds =[]
model.eval()
with torch.inference_mode():
    for X,y in test_dataloader:
        X = X.to(device)
        y_logits = model(X)
        y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
y_pred_tensor = torch.cat(y_preds)
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
)
plt.show()
