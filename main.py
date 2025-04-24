from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, json
from datetime import datetime
from utils import ingest_files

app = FastAPI()

UPLOAD_DIR = "uploads"
METADATA_DIR = "metadata"

# Create folders if not exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    employee_name: str = Form(...),
    employee_role: str = Form(...),
    employee_department: str = Form(...),
    document_type: str = Form(...),
    uploader: str = Form(...),
    access_roles: str = Form(...),  # comma-separated
    tags: str = Form(...),
    summary: str = Form(...),
    confidentiality_level: str = Form(...)
):
    # Save file to uploads/
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Build metadata
    metadata = {
        "filename": file.filename,
        "file_type": file.content_type.split("/")[-1],
        "document_type": document_type,
        "uploader": uploader,
        "employee_name": employee_name,
        "employee_role": employee_role,
        "employee_department": employee_department,
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "access_roles": [role.strip() for role in access_roles.split(",")],
        "tags": [tag.strip() for tag in tags.split(",")],
        "summary": summary,
        "confidentiality_level": confidentiality_level
    }

    # Save metadata as .json
    metadata_path = os.path.join(METADATA_DIR, f"{file.filename}.json")
    with open(metadata_path, "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)

    return {"message": "File and metadata saved successfully."}

@app.post("/ingest")
def ingest_documents(): 
    ingest_files()
    return {"status": "success", "message": "All documents embedded and indexed"}

