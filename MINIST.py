import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"


# train_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
# test_data = datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
train_data = datasets.EMNIST(
    root="data",
    train=True,
    download=True,
    split="byclass",
    transform=ToTensor(),
)
test_data = datasets.EMNIST(
    root="data",
    train=False,
    download=True,
    split="byclass",
    transform=ToTensor()
)
print(len(train_data), len(test_data))
test_dataloader = DataLoader(
    test_data,
    batch_size=128,
    #batch_size = 32,
    shuffle=False
)
train_dataloader = DataLoader(
    train_data,
    batch_size=128,
    #batch_size = 32,
    shuffle=True
)
print(len(train_data), len(test_data))
class_names = train_data.classes
from helper_functions import accuracy_fn
def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss+=loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred= y_pred.argmax(dim=1))
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
                device: torch.device = device):
     test_loss, test_acc = 0, 0
     model.eval()
     with torch.inference_mode():
          for batch, (X,y) in enumerate(dataloader):
                X,y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss+=loss
                test_acc += accuracy_fn(y_true=y,
                                        y_pred= y_pred.argmax(dim=1))
          test_loss /= len(dataloader)
          test_acc /= len(dataloader)
          print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")

class MNISTModel(nn.Module):
    def __init__(self, input_shape: int,hidden_units: int, output_shape: int):
        super().__init__()
        self.layer1 = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.layer2 = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape))
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.classifier(x)
        return x

model = MNISTModel(input_shape=1,output_shape=len(train_data.classes),hidden_units=15).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                               lr=0.01)

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}")
    train_step(model=model,
               dataloader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)
    test_step(model=model,
              dataloader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
y_preds =[]
model.eval()
with torch.inference_mode():
    for X,y in test_dataloader:
        X = X.to(device)
        y_logits = model(X)
        y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
y_pred_tensor = torch.cat(y_preds)
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10,7)
)
plt.show()