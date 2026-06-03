from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import tempfile
import os
import re

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

def clean_text(text):
    # Normalizar saltos de línea
    text = text.replace("\r", "")

    # Eliminar espacios repetidos
    text = re.sub(r"[ \t]+", " ", text)

    # Eliminar demasiados saltos seguidos
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Eliminar líneas que solo contienen números de página
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    return text.strip()

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

        text = clean_text(text)

        return {
            "pages": len(reader.pages),
            "characters": len(text),
            "text": text
        }

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)