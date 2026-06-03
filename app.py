from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import tempfile
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIZE = 200 * 1024 * 1024  # 200 MB


@app.get("/")
def root():
    return {"status": "ok"}


def clean_text(text):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    temp_path = None

    try:
        size = 0

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            temp_path = tmp.name

            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_SIZE:
                    tmp.close()
                    os.remove(temp_path)

                    return JSONResponse(
                        {"error": "PDF supera 200 MB"},
                        status_code=413
                    )

                tmp.write(chunk)

        reader = PdfReader(temp_path)

        text_parts = []

        for page in reader.pages:
            text_parts.append(page.extract_text() or "")

        text = clean_text("\n".join(text_parts))

        return {
            "pages": len(reader.pages),
            "characters": len(text),
            "size_mb": round(size / (1024 * 1024), 2),
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