import os
from flask import Flask,request,redirect,url_for,send_from_directory
import numpy as np
import torchvision.models as models
import torchvision
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms

UPLOAD_FOLDER = 'temp/data/uploads'
MODEL_CHECKPOINT = 'temp/checkpoint/ramen.pth'
CSS_URL = "https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css"
HTML_HEADER = f"""
<!DOCTYPE html><html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="{CSS_URL}"><body>
<section class="hero has-background-info"><div class="hero-body">
    <h1 class="title">ラーメン判定AI</h1></div></section>
"""
image_size = 25

preprocess = transforms.Compose([
     transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),

])


LABELS = ["味噌ラーメン","塩ラーメン","冷やし中華","担々麺"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 4)
model.load_state_dict(torch.load(MODEL_CHECKPOINT))

@app.route('/')
def root():
    return f"""{HTML_HEADER}
    <div class="box file">
        <form method="post" action="/predict"
            enctype="multipart/form-data">
            <input type="file" name="file" class="file-label" /><br>
            <input type="submit" value="画像判定"
                class="button is-primary" />
        </form></div>
    </body></html>
    """

@app.route('/predict',methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if not (file and file.filename.endswith(('.jpg', '.jpeg'))):
        return f"""{HTML_HEADER}
        <h1>アップロードできるのは画像のみです</h1></body></html>"""
    # 画像をディレクトリに保存 --- (※6)
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    # 画像を読み込む --- (※7)
    try:
        img = Image.open(file_path).convert('RGB') # Open image and convert to RGB
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
    except Exception as e:
        return f"""{HTML_HEADER}
        <h1>画像を読み込めませんでした</h1></body></html>"""
# 予測を行う --- (※8)
    model.eval()
    with torch.no_grad(): # 勾配計算を無効化 (推論時)
        output = model.forward(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0) # 確率に変換
        predicted_index = torch.argmax(probabilities).item() # 最も高い確率のインデックスを取得
        predicted_probability = probabilities[predicted_index].item() * 100 # 予測確率をパーセンテージで取得

    return f"""{HTML_HEADER}
    <div class="card" style="font-size:2em; padding:1em;">
        判定結果: {LABELS[predicted_index]} (精度:{predicted_probability:.2f}%)<br>
        <img src="upload/{filename}" width="400"><br>
        <a href="/" class="button">次の画像</a>
    </div></body></html>
    """
# アップロードされたファイルを返す --- (※11)
@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8888)

