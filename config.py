# config.py

from pathlib import Path

# Base directory for uploaded documents
UPLOAD_DIR = Path("uploads")
VECTOR_DB_DIR = Path("vector_store")

# You can also define other constants here if needed
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
