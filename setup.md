# Streamlit Photo Processor - Deployment Guide

## Quick Start (Local Testing)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app locally:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   - Open your browser to `http://localhost:8501`
   - Default password: `grassroots2024`

## Deploying to Streamlit Cloud (FREE & EASIEST)

### Step 1: Prepare Your Files
Create a folder with these 3 files:
- `app.py` (the main Streamlit app)
- `requirements.txt` (dependencies)
- `README.md` (optional description)

### Step 2: Push to GitHub
1. Create a new repository on GitHub
2. Upload your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path:** `app.py`
   - **Python version:** 3.11
6. Click "Deploy"!

Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Step 4: Set Up Password (Environment Variable)
**IMPORTANT:** Set your password as a secret in Streamlit Cloud:

1. Go to your app dashboard on Streamlit Cloud
2. Click on "⚙️ Settings" → "Secrets"
3. Add this to your secrets:
   ```toml
   password = "your_secure_password_here"
   ```
4. Click "Save"

The app will automatically use this password. It's never visible in your source code!

---

## Alternative: Deploy to Other Platforms

### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```
2. Deploy via Heroku CLI or GitHub integration

### Railway
1. Connect your GitHub repo
2. Railway auto-detects Streamlit apps
3. Set start command: `streamlit run app.py`

### Google Cloud Run
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8080
   CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
   ```
2. Deploy with `gcloud run deploy`

---

## Usage Instructions for Your Users

1. **Login:** Enter the password you've shared
2. **Upload watermark:** First, upload your badge.png or logo
3. **Upload photos:** Select multiple photos to process
4. **Process:** Click the "Process All Images" button
5. **Download:** 
   - Download all as a ZIP file, or
   - Preview and download individual images

---

## Security Notes

- The password in the code is **basic protection** only
- For production use with sensitive images, consider:
  - Using Streamlit's secrets management
  - Adding proper authentication (OAuth, etc.)
  - Implementing HTTPS
  - Setting up rate limiting

### Using Streamlit Secrets (Recommended)
1. Create `.streamlit/secrets.toml`:
   ```toml
   password = "your_secure_password_here"
   ```
2. In `app.py`, change:
   ```python
   CORRECT_PASSWORD = st.secrets["password"]
   ```
3. Add `.streamlit/` to `.gitignore`
4. Set the secret in Streamlit Cloud dashboard

---

## Troubleshooting

**App crashes on image processing:**
- Check image format is supported (JPG, PNG)
- Ensure images have detectable objects
- Try with smaller images first

**Memory errors:**
- Streamlit Cloud has 1GB RAM limit
- Process fewer images at once
- Reduce image dimensions if needed

**Slow processing:**
- OpenCV operations are CPU-intensive
- Consider upgrading to paid hosting for better performance
- Process images in smaller batches

---

## Support

For issues or questions, contact the repository maintainer.