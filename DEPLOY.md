# Deployment Guide - Temples of Tamil Nadu Recognition

This project is ready for deployment. Below are the recommended platforms and steps.

## Option 1: Streamlit Community Cloud (Recommended)
This is the easiest way to deploy for free.

### Steps:
1.  **Push to GitHub**: Create a repository and push your project files (including `temple_mlp.pt`, `temples_metadata.json`, `app.py`, `backend.py`, and `requirements.txt`).
2.  **Connect to Streamlit**:
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Sign in with GitHub.
    *   Click "New app".
    *   Select your repo, branch, and main file (`app.py`).
3.  **Advanced Settings**:
    *   If the app fails to find system libraries for OpenCV, add a file named `packages.txt` to your repo with the following content:
        ```text
        libgl1-mesa-glx
        libglib2.0-0
        ```

## Option 2: Hugging Face Spaces
Excellent for AI/ML projects and provides a persistent URL.

### Steps:
1.  **Create Space**: Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
2.  **Select SDK**: Choose **Streamlit**.
3.  **Upload Files**: Upload all project files to the repository.
4.  **Automatic Setup**: Hugging Face will automatically detect `requirements.txt` and install everything.

## Option 3: Docker Deployment (Cloud Run / AWS / GCP)
Use this if you want to deploy on your own server or a container service.

### 1. Build Image:
```bash
docker build -t temple-recognition-app .
```

### 2. Run Container:
```bash
docker run -p 8501:8501 temple-recognition-app
```

---

## Important Deployment Files
- `requirements.txt`: Lists all Python libraries.
- `packages.txt`: (Optional) Lists system-level libraries for Streamlit Cloud.
- `temple_mlp.pt`: **MUST** be included in your deployment to enable Version 5.
- `temples_metadata.json`: The core data source.
