from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import requests
import io
import os

app = FastAPI()

# تحميل النموذج من Google Drive إذا لم يكن موجودًا
model_path = "alzheimers_hybrid_model.pt"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1_0BKg57f-pQKjZHLMx6A6tUN1y_xHZKD"
    response = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(response.content)

# تحميل النموذج إلى الذاكرة
device = torch.device("cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# التحويلات المطلوبة للصورة
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# أسماء الفئات حسب التدريب
class_names = ["AD", "CN", "MCI"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        label = class_names[predicted.item()]
        return JSONResponse(content={"diagnosis": label})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
