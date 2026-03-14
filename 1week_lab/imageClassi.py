import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# -------------------------------
# 1. Image Preprocessing
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((64,64)),   # resize images
    transforms.ToTensor()         # convert to tensor
])

# -------------------------------
# 2. Load Dataset
# -------------------------------

dataset = ImageFolder("dataset", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

print("Total images:",len(dataset))
print("Training images:",train_size)
print("Testing images:",test_size)

# -------------------------------
# 3. Logistic Regression Model
# -------------------------------

class LogisticRegressionModel(nn.Module):

    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = nn.Linear(3*64*64,1)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = torch.sigmoid(self.linear(x))
        return x


model = LogisticRegressionModel()

# -------------------------------
# 4. Loss and Optimizer
# -------------------------------

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

# -------------------------------
# 5. Training
# -------------------------------

epochs = 10

train_losses = []
test_losses = []

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for images,labels in train_loader:

        labels = labels.float().view(-1,1)

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    train_losses.append(train_loss)

    # -------------------------------
    # Testing Loss
    # -------------------------------

    model.eval()
    test_loss = 0

    with torch.no_grad():

        for images,labels in test_loader:

            labels = labels.float().view(-1,1)

            outputs = model(images)

            loss = criterion(outputs,labels)

            test_loss += loss.item()

    test_loss = test_loss/len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

# -------------------------------
# 6. Plot Training vs Test Loss
# -------------------------------

plt.figure()

plt.plot(train_losses,label="Training Loss")
plt.plot(test_losses,label="Test Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Test Loss")
plt.legend()

plt.show()

# -------------------------------
# 7. Confusion Matrix Values
# -------------------------------

TP = 0
TN = 0
FP = 0
FN = 0

model.eval()

with torch.no_grad():

    for images,labels in test_loader:

        outputs = model(images)

        predictions = (outputs >= 0.5).float()

        labels = labels.view(-1,1)

        for p,l in zip(predictions,labels):

            if p == 1 and l == 1:
                TP += 1

            elif p == 0 and l == 0:
                TN += 1

            elif p == 1 and l == 0:
                FP += 1

            elif p == 0 and l == 1:
                FN += 1


# -------------------------------
# 8. Print Results
# -------------------------------

print("\nConfusion Matrix Values")

print("True Positive (TP):",TP)
print("True Negative (TN):",TN)
print("False Positive (FP):",FP)
print("False Negative (FN):",FN)