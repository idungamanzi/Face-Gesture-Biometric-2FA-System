# Face-Gesture-Biometric-2FA-System
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
