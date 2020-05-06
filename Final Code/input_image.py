from PIL import Image

# open the image
input_img = Image.open("Digit6.png",str = "r") # the image is changable
# Process the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
input_tensor = preprocess(input_img)
input_batch = input_tensor.unsqueeze(0) # create a mini batch s expect by the model

# model is just the one from the CNN or whatever net it is.
model = CNN()
model.eval()


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
