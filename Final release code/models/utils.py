import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color = 'blue')
    plt.legend(['value'], loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation= 'none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth = 10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim = 1, index = idx, value = 1)
    return out


def infer(model, inputs):
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    x_batch = Variable(inputs, volatile=True, requires_grad=False)

    scores = model(x_batch)

    scores, predicted = torch.max(scores, 1)

    return predicted

# Visualize some samples
def visualize(test_loader):
    batch = next(iter(test_loader))
    samples = batch[0][:5]
    y_true = batch[1]
    for i, sample in enumerate(samples):
        plt.subplot(1, 5, i+1)
        plt.title('Numbers: %i' % y_true[i])
        plt.imshow(sample.numpy().reshape((28, 28)))
        plt.axis('off')

from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms

def image_loader(filename):
    # load the image and return the cuda tensor
    image = Image.open(filename)
    loader = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    image = loader(image).float()
    image = Variable(image,requires_grad=True)
    image = image.unsqueeze(0)
    return image
