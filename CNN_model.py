# importación de librerias
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt

# descargar datos
data = '/media/josh/MyData2SSD/Databases/cifar-10-batches-py'
n_train = 50000
n_val = 5000
n_test = 5000
m_size = 16

# transformación de datos
transform_cifar = T.Compose([
                T.ToTensor(),
                T.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
            ])

# datos de entrenamiento
cifar10_train = datasets.CIFAR10(data, train=True, download=True,
                             transform=transform_cifar)
train_loader = DataLoader(cifar10_train, batch_size=m_size, 
                          sampler=sampler.SubsetRandomSampler(range(n_train)))
# datos de validación
cifar10_val = datasets.CIFAR10(data, train=False, download=True,
                           transform=transform_cifar)
val_loader = DataLoader(cifar10_val, batch_size=m_size, 
                        sampler=sampler.SubsetRandomSampler(range(n_val)))
# datos de prueba
cifar10_test = datasets.CIFAR10(data, train=False, download=True, 
                            transform=transform_cifar)
test_loader = DataLoader(cifar10_test, batch_size=m_size,
                        sampler=sampler.SubsetRandomSampler(range(n_val, len(cifar10_test))))
                        
                        
# uso de gpu
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


# función train
def train(model, optimiser, epochs=100):
    model = model.to(device=device)
    for epoch in range(epochs):
        for i, (xi, yi) in enumerate(train_loader):
            model.train()
            xi = xi.to(device=device, dtype=torch.float32)
            yi = yi.to(device=device, dtype=torch.long)
            scores = model(xi)
            loss = F.cross_entropy(input= scores, target=yi)
            optimiser.zero_grad()           
            loss.backward()
            optimiser.step()              
        acc = accuracy(model, val_loader) 
        print(f'Epoch: {epoch}, loss: {loss.item()}, accuracy: {acc},')
 

# funcion accuracy
def accuracy(model, loader):
    num_correct = 0
    num_total = 0
    model.eval()
    model = model.to(device=device)
    with torch.no_grad():
        for xi, yi in loader:
            xi = xi.to(device=device, dtype = torch.float32)
            yi = yi.to(device=device, dtype = torch.long)
            scores = model(xi) 
            _, pred = scores.max(dim=1) 
            num_correct += (pred == yi).sum() 
            num_total += pred.size(0)
        return float(num_correct)/num_total  
        

# diseño del modelo
nodo1 = 128
nodo = 64
lr = 0.001
epochs = 100
model = nn.Sequential(nn.Flatten(),
                       nn.Linear(in_features=32*32*3, out_features=nodo1), nn.ReLU(),
                       nn.Linear(in_features=nodo1, out_features=nodo), nn.ReLU(),
                       nn.Linear(in_features=nodo, out_features=10), nn.Softmax())

optimiser = torch.optim.SGD(model.parameters(), lr=lr)

# entrenar el modelo
train(model, optimiser, epochs)


# curva loss datos de entrenamiento
def loss(model, num_epoch):
    loss_values = []
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data                   
            scores = model(images)
            loss = F.cross_entropy(input= scores, target=labels)
            optimiser.zero_grad()           
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * images.size(0) 
        epoch_loss = running_loss / len(train_loader)
        loss_values.append(epoch_loss)
    plt.plot(loss_values, 'b', label='Training loss')

    
# curva loss datos de validación
def loss_2(model, num_epoch):
    loss_values = []
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            images, labels = data                        
            scores = model(images)
            loss = F.cross_entropy(input= scores, target=labels)
            optimiser.zero_grad()           
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(val_loader)
        loss_values.append(epoch_loss)
    plt.plot(loss_values, 'r', label='Validation loss')
                        
                        
# grafica Training & Validation loss
loss = loss(model, 100)
val_loss = loss_2(model, 100)
plt.title('Training & Validation loss')                        
                        
                        
                        
