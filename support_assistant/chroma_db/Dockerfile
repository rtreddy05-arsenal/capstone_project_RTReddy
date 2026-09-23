# Use a lightweight Python base image
FROM python:3.10-slim

# Hugging Face Spaces requires running as a non-root user with UID 1000
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set environment variables for the user's home and PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements first to leverage Docker layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and the local ChromaDB folder
COPY --chown=user . .

# Expose port 7860 for local testing and Hugging Face compatibility
EXPOSE 7860

# Start the FastAPI server (change "main:app" to "app:app" if your file is named app.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]