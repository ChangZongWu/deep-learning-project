import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn

from CostumeDatasets import device
from helper_functions import plot_decision_boundary

X,y = make_blobs(n_samples=1000, n_features=2,centers=5)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train,X_test, y_train, y_test = train_test_split(X,
                                      y,
                                        test_size=0.2)
# plt.figure(figsize=(10, 7))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

class MultiClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_1 = nn.Linear(in_features=2,out_features=10)
        self.linear_layer_2 = nn.Linear(in_features=10,out_features=10)
        self.linear_layer_3 = nn.Linear(in_features=10,out_features=5)
        self.relu = nn.ReLU()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.linear_layer_3(self.relu(self.linear_layer_2(self.relu(self.linear_layer_1(x)))))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiClassModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                             lr=0.1)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
epochs = 2000
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits,y_train.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.inference_mode():
        model.eval()
        test_logits = model(X_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits,y_test.long())
    if epoch%200 == 0:
        print(f"Epoch: {epoch} | "
              f"Train loss: {loss:.5f} | "
              f"Test loss: {test_loss:.5f} ")

# model.eval()
# with torch.inference_mode():
#     y_logits = model(X_test)
# y_pred_probs = torch.softmax(y_logits, dim=1)
# print(y_pred_probs)
# y_pred_classes = y_pred_probs.argmax(dim=1)
# print(y_pred_classes)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()