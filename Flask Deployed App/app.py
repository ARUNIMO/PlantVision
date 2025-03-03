import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.autograd import Variable

# Load disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load Tamil disease information
tamil_disease_info = pd.read_csv('disease_info_tamil.csv', on_bad_lines='skip')


# Model loading using ResNet-50
class ResNet50(torch.nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = torch.nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = torch.nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = torch.nn.Sequential(
            torch.nn.MaxPool2d(4),
            torch.nn.Flatten(),
            torch.nn.Linear(512, num_diseases)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              torch.nn.BatchNorm2d(out_channels),
              torch.nn.ReLU(inplace=True)]
    if pool:
        layers.append(torch.nn.MaxPool2d(4))
    return torch.nn.Sequential(*layers)

# Load the trained model
model = ResNet50(3, 38)
model.load_state_dict(torch.load("plant-disease-model_1.pth"))
model.eval()

def transform_image(image_path):
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def prediction(image_path):
    input_data = transform_image(image_path)
    input_data = Variable(input_data).to('cuda')
    model.to('cuda')
    output = model(input_data)
    output = output.detach().cpu().numpy()
    index = np.argmax(output)
    return index

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')
pred1=0
def setpred(self,pred):
    self.pred1=pred
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        file=file_path
        image.save(file_path)
        pred = prediction(file_path)
        global pred1
        pred1=pred
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/submit_tamil', methods=['GET', 'POST'])
def submit_tamil():
    if request.method == 'POST':
        pred = pred1
        title_tamil = tamil_disease_info['disease_name'][pred]
        description_tamil = tamil_disease_info['description'][pred]
        prevent_tamil = tamil_disease_info['Possible Steps'][pred]
        image_url_tamil = tamil_disease_info['image_url'][pred]
        return render_template('submit_tamil.html', title=title_tamil, desc=description_tamil,
                               prevent=prevent_tamil, image_url=image_url_tamil, pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
