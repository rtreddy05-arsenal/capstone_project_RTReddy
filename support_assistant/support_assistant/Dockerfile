FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and the ChromaDB vector store
COPY app.py .
COPY chroma_db/ ./chroma_db/

# Expose port 7860 (compatible with local use and Hugging Face Spaces)
EXPOSE 7860

# Serve the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

