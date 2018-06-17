from io import BytesIO
from PIL import Image

def image_loader(image_base64):
	image_bytes  = BytesIO(image_base64)

	image = Image.open(image_bytes)
	image = loader(image).float()
	image = Variable(image, requires_grad=True)
	image = image.unsqueeze(0)

	return image.float()
