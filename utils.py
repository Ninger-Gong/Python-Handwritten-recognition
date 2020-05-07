import torch
from matplotlib import pyplot as plt

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
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim = 1, index = idx, value = 1)
    return out

def input_image(filename):
    input_img = Image.open((filename,"r"))
    # preprocess the picture
    preprocess = transforms.Compose( [
        transforms.Resize( 256 ),
        transforms.CenterCrop( 224 ),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
    ] )
    input_tensor = preprocess( input_img )
    input_batch = input_tensor.unsqueeze( 0 )
    return input_batch
