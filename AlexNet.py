'''
Implementation of AlexNet Architecture from scratch in PyTorch.
Trained on CIFAR10 Dataset.
Model params: 0.316 M
Model Performance(Best Achieved):
    Epoch 4:        train_loss: 0.5804058068590484  test_loss:  0.7785483202534521
'''




# Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy

# Configuration
epochs = 20
batch_size = 60
lr1 = 1e-3
lr2 = 1e-4

# Data Collection
train_dataset = datasets.CIFAR10(root = 'datasets/', download = True, train = True, transform = transforms.ToTensor())
test_dataset = datasets.CIFAR10(root = 'datasets/', download = True, train = False, transform = transforms.ToTensor())

# Data Preprocessing
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

def show(img):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()
img, _ = next(iter(train_dataloader))
#show(torchvision.utils.make_grid(img))        
    
# Model Building
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3),  # 30*30*32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # 15*15*32
            
            nn.Conv2d(32, 64, kernel_size = 3),  # 13*13*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                  # 6*6*64
            
            nn.Flatten(),
            nn.Linear(6*6*64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x, targets = None):
        logits = self.cnn(x)    # B, 10
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def classify(self, x):
        logits, _ = self(x)
        probs = F.softmax(logits, dim = -1)
        ix = torch.multinomial(probs, num_samples = 1)
        return ix
    
model = CNN()
print(f"Model Parameters: {sum([p.nelement() for p in model.parameters()])/1e6:.2f} M")

# Model Training
train_accuracy = MulticlassAccuracy(num_classes = 10)
test_accuracy = MulticlassAccuracy(num_classes = 10)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr1)
for epoch in range(epochs):
    if epoch > 9:
        for param in optimizer.param_groups:
           param['lr'] = lr2
    losses = []
    for x, y in train_dataloader:
        logits, loss = model(x, y)
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
        train_accuracy.update(logits, y)
    print(f"Epoch {epoch+1}:\tacc: {train_accuracy.compute().item():.2f}\ttrain_loss: {sum(losses)/len(losses):.2f}", end = "\t")
    losses = []
    model.eval()
    for x, y in test_dataloader:
        logits, loss = model(x, y)
        losses.append(loss.item())
        test_accuracy.update(logits, y)
    print(f"test_loss: {sum(losses)/len(losses):.2f}\tacc: {test_accuracy.compute().item():.2f}")
    model.train()
    
        

