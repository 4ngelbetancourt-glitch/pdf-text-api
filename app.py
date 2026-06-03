from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pypdf import PdfReader
import tempfile
import os

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    temp_path = None

    try:
        content = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        reader = PdfReader(temp_path)

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        return {"text": text}

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)