import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms

from model import LeNet5

#define relevenat variables for ML task
batch_size = 64
num_class = 10
learning_rate = 0.001
num_epochs = 1

#select device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current deivce: {}'.format(device))

#Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(root = '/media/data/cuonghv14/TuyenDT4/learn_deeplearning/datasets',
                                           train = True,
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                           download = False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)

model = LeNet5(num_class).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #Forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        #Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
saved_weights = './trained_weight/weights.pt'
torch.save(model.state_dict(), saved_weights)