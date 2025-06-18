import torch
from torch import nn
from sklearn.datasets import make_circles

from helper_functions import plot_decision_boundary

X,y = make_circles(n_samples=1000,
                   noise=0.03,
                   random_state=None)
import matplotlib.pyplot as plt
# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.show()
from sklearn.model_selection import train_test_split
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train,X_test, y_train, y_test = train_test_split(X,
                                      y,
                                        test_size=0.2)
print(len(X_train), len(X_test), len(y_train), len(y_test))

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_1 = nn.Linear(in_features=2,out_features=10)
        self.linear_layer_2 = nn.Linear(in_features=10,out_features=10)
        self.linear_layer_3 = nn.Linear(in_features=10,out_features=1)
        self.relu = nn.ReLU()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.linear_layer_3(self.relu(self.linear_layer_2(self.relu(self.linear_layer_1(x)))))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CircleModel().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                             lr=0.1)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
epochs = 1000
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    y_pred= torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.inference_mode():
        model.eval()
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,y_test)
    if epoch%100 == 0:
        print(f"Epoch: {epoch} | "
              f"Train loss: {loss:.5f} | "
              f"Test loss: {test_loss:.5f} ")

# plt.figure(figsize=(12, 6))
# plt.title("Test")
# plot_decision_boundary(model, X_test, y_test)
# plt.show()
