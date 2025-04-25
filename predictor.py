
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

from io import BytesIO


class Predictor:
  def __init__(self, model, transforms):
    self.count = 0
    self.model = model
    self.preprocessor =  transforms

  def classify(self, image_payload_bytes):
    pil_image = Image.open(BytesIO(image_payload_bytes))
    class_labels = ['H1', 'H2', 'MO', 'SI']

    pil_images = [pil_image]  #batch size is one
    input_tensor = torch.cat(
        [self.preprocessor(i).unsqueeze(0) for i in pil_images])
    self.model.eval()
    with torch.no_grad():
        output_tensor = self.model(input_tensor)
    _,preds = torch.max(output_tensor, 1)
    predicted_label = [class_labels[i] for i in preds]
    return {"label": predicted_label}
    # return {"class_index": int(torch.argmax(output_tensor[0]))}
