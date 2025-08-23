# Face-Gesture-Biometric-2FA-System
## Quick Start
```bash
# (Recommended) create & activate a virtual environment for Python 3.11
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install system deps (Linux): dlib build tools may be needed
# e.g., Ubuntu: sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev

# Install Python libs
pip install -r requirements.txt

# Run
python access_system.py
```

## Notes
- A `secret.key` is created on first run for Fernet encryption.
- Encrypted blobs are stored under `data/user_<id>/` and paths are indexed in `users.db` (SQLite).
- Press `q` in any camera window to abort current capture.
- Face embeddings use HOG model for portability; for speed, consider CNN (requires CUDA).

## Menu
- **Enroll new user** → captures multiple face embeddings, then multiple gesture samples, storing encrypted blobs and raw frames.
- **Verify (auto)** → detects which enrolled user is presenting via face match, then asks for gesture and validates with DTW.
- **List users** / **Delete user** → basic management.

## Tuning
- Adjust `FACE_TOLERANCE`, `GESTURE_SECONDS`, `GESTURE_FPS`, `GESTURE_SAMPLES`, and `DTW_SAKOE` in the script for your setup.
