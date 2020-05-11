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


# move the input and model to GPU for speed if available
input_batch = image_loeader(filename)
if torch.cuda.is_available():
    input_batch = input_batch.to(Device)
