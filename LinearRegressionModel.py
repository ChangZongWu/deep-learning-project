import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import device
from pathlib import Path

weight = 0.5
bias = 0.5
start = 0
end =1
step = 0.01
X = torch.arange(start, end, step)
print(X.shape)
y= weight * X + bias
print(X[:10])
print(y[:10])
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                        test_data = X_test,
                        test_labels = y_test,
                        predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label = "Train data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()

#plot_predictions()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                dtype=torch.float),
                                    requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,
                                             dtype=torch.float),
                                 requires_grad=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

model_0 = LinearRegressionModel()
print(model_0.state_dict())
with torch.inference_mode():
    y_preds = model_0(X_test)
print(len(X_test))
print(len(y_preds))

# plot_predictions(predictions=y_preds)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)
epochs = 1000
train_loss_values = []
test_loss_values = []
epoch_count = []
for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred,y_test.type(torch.float))
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

print(model_0.state_dict())
print(f"weights: {weight}, bias: {bias}")
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
#plot_predictions(predictions=y_preds)
from pathlib import Path
Model_Path = Path("models")
Model_Path.mkdir(parents=True, exist_ok=True)
Model_Name = "LinearRegressionModel_0.pth"
Model_Save_Path = Model_Path / Model_Name
torch.save(model_0.state_dict(),f=Model_Save_Path)

load_model_0 = LinearRegressionModel()
load_model_0.load_state_dict(torch.load(Model_Save_Path))
load_model_0.eval()
with torch.inference_mode():
    load_y_preds = load_model_0(X_test)
print(load_y_preds==y_preds)