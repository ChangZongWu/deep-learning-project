import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

from CostumeDatasets import train_data

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.001
X = torch.arange(start, end, step).unsqueeze(dim=1)
print(X.shape)
y = weight * X + bias
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:],y[train_split:]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(len(X_train),len(y_train),len(X_test),len(y_test))
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
        self.linear = nn.Linear(in_features=1,
                                out_features=1)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.linear(x)

model = LinearRegressionModel().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 1000
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.inference_mode():
        model.eval()
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred,y_test)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# plot_predictions(train_data = X_train.cpu(),
#                  train_labels = y_train.cpu(),
#                  test_data = X_test.cpu(),
#                  test_labels = y_test.cpu(),
#                  predictions = test_pred.cpu())
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "linear_regression_model_gpu.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)

loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model.to(device)
model.eval()
with torch.inference_mode():
    y_preds = model(X_test)
loaded_model.eval()
with torch.inference_mode():
    loaded_model_pred = loaded_model(X_test)
print(y_preds == loaded_model_pred)