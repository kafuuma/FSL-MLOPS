from fastapi import FastAPI, File, UploadFile
from model import CNN
from predictor import Predictor
import torchvision.transforms as transforms
import torch


app = FastAPI()

@app.get('/')
def main():
    return {'message':'Hyperspectral Imaging'}

@app.post('/classify')
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    model = model = CNN()
    model.load_state_dict(torch.load('./FSL.pth'))
    model.eval()
    classfier = Predictor(model=model, transforms=transform)
    output = classfier.classify(image_bytes)
    print(output)
    return output