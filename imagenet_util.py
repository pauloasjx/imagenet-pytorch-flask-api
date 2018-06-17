
from PIL import Image

def image_loader(image_file):
    image = Image.open(image_file)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)

    return image.float()
