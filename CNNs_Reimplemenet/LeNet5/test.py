import torch
import torchvision 
import torchvision.transforms as transforms
from model import LeNet5

batch_size = 64
num_classes = 10
saved_weights = './trained_weight/weights.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = torchvision.datasets.MNIST(root = '/media/data/cuonghv14/TuyenDT4/learn_deeplearning/datasets',
                                          train = False,
                                          transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                          download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = LeNet5(num_classes)
model.load_state_dict(torch.load(saved_weights))
model.eval()
model.to(device)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
	 