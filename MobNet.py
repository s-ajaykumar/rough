'''
Implementation of MobileNetV1 Architecture from scratch in PyTorch.
Trained on CIFAR10 Dataset.
Model params: 0.05 M
Model Performance(Best Achieved):
    Epoch 11:       acc: 0.76       train_loss: 0.68        test_loss: 0.85     acc: 0.70
'''




import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy


# Configuration
epochs = 20
batch_size = 10
lr1 = 1e-3
lr2 = 1e-4


# Data Collection
train_dataset = datasets.CIFAR10(root = 'datasets/', download = True, train = True, transform = transforms.ToTensor())
test_dataset = datasets.CIFAR10(root = 'datasets/', download = True, train = False, transform = transforms.ToTensor())


# Data Preprocessing
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)


# Model Building
class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups = out_channels),                  
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),                           
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)
        
class MobileNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            DepthWiseConv(in_channels, in_channels, 3, 1),                      # 30*30*32
            PointWiseConv(in_channels, 32, 1),                                  # 30*30*32

            DepthWiseConv(32, 32, 3, 1),                                        # 28*28*32      
            PointWiseConv(32, 64, 1),                                           # 28*28*64  
            
            DepthWiseConv(64, 64, 3, 1),                                        # 26*26*64   
            PointWiseConv(64, 48, 1),                                           # 26*26*48  
            
            DepthWiseConv(48, 48, 3, 1),                                        # 24*24*48   
            PointWiseConv(48, 128, 1),                                          # 24*24*128
            
            DepthWiseConv(128, 128, 3, 1),                                      # 22*22*48   
            PointWiseConv(128, 84, 1),                                          # 22*22*84
            
            DepthWiseConv(84, 84, 3, 1),                                        # 20*20*84   
            PointWiseConv(84, 256, 1),                                          # 20*20*256
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x, targets = None):
        logits = self.net(x)    # B, 10
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def classify(self, x):
        logits, _ = self(x)
        probs = F.softmax(logits, dim = -1)
        ix = torch.max(probs, dim = -1, keepdim = True)
        return ix
    
model = MobileNet(in_channels = 3)
print(f"Model Parameters: {sum([p.nelement() for p in model.parameters()])/1e6:.2f} M")


# Model Training and Validation
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
    train_accuracy.reset()
    losses = []
    model.eval()
    for x, y in test_dataloader:
        logits, loss = model(x, y)
        losses.append(loss.item())
        test_accuracy.update(logits, y)
    print(f"test_loss: {sum(losses)/len(losses):.2f}\tacc: {test_accuracy.compute().item():.2f}")
    test_accuracy.reset()
    model.train()