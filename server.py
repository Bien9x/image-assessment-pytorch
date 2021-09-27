from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import predict
from typing import List
from PIL import Image
import base64
from io import BytesIO

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = Image.open(file.file)
    score = predict.predict_image(contents, "resnet18", "cp/checkpoint_ava_resnet18.pt")
    return {"filename": file.filename, "score": score}


@app.post("/uploadfiles/")
async def create_upload_files(request: Request, files: List[UploadFile] = File(...)):
    # contents = [Image.open(file.file) for file in files]

    contents = []
    img_strs = []
    for file in files:
        img = Image.open(file.file)
        contents.append(img)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_strs.append(img_str)
    scores = predict.predict_images(contents, "resnet18", "cp/checkpoint_ava_resnet18.pt")
    items = [{'src': src, 'score': score} for src, score in zip(img_strs, scores)]
    items.sort(key=lambda x: x['score'])
    return templates.TemplateResponse("scores.html", {"request": request, "items": items})


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    content = """
<body>
<form method="post" action="/uploadfiles/" enctype="multipart/form-data">
    <input type="file" id="file_upload" name="files" accept="image/*" multiple>
    <button class="btn btn-outline-secondary" type="submit">upload</button>
</form>
</body>
    """
    # return HTMLResponse(content=content)
    return templates.TemplateResponse("index.html", {"request": request})
