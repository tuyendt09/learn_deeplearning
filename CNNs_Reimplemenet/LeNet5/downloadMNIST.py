'''
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
'''
import torchvision
import torchvision.transforms as transforms


#Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(root = '/media/data/cuonghv14/TuyenDT4/learn_deeplearning/datasets',
                                           train = True,
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                           download = False)
img = train_dataset.__getitem__(1)[0]
print(img.size())


# test_dataset = torchvision.datasets.MNIST(root = '/media/data/cuonghv14/TuyenDT4/learn_deeplearning/datasets',
#                                           train = False,
#                                           transform = transforms.Compose([
#                                                   transforms.Resize((32,32)),
#                                                   transforms.ToTensor(),
#                                                   transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
#                                           download=False)

