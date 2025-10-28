# Face-Gesture-Biometric-2FA-System
A high-accuracy, two-factor authentication system that combines facial recognition and gesture recognition for secure access control. This system runs locally using a webcam and provides enterprise-grade biometric security.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/opencv-4.5%2B-red.svg)](https://opencv.org/)

## 🌟 Features

### 🎯 High-Accuracy Authentication
- Facial Recognition: Multi-sample enrollment with averaged embeddings for robust face profiles.
- Gesture Recognition: Advanced hand landmark tracking with 100+ feature points per gesture.
- Dual-Factor Security: Access granted only when both face AND gesture verification pass.
- Anti-Spoofing: Strict single-face detection and quality checks prevent unauthorized access.

### 🔒 Security
- End-to-End Encryption*: All user profiles encrypted using Fernet symmetric encryption.
- Tamper-Resistant: Encrypted storage prevents profile manipulation.
- Privacy-First: All processing happens locally — no cloud, no data transmission.
- Secure Key Management: Auto-generated encryption keys stored separately.

### 🚀 Advanced Technology
- Face Recognition: Uses `face_recognition` (dlib-based deep learning embeddings).
- Hand Tracking: MediaPipe Hands for precise 21-landmark hand detection.
- Multi-Metric Verification: 
  - Euclidean distance for face matching  
  - Cosine similarity for gesture comparison  
  - Quality scoring for capture validation

---

## 📋 Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [How It Works](#how-it-works)
- [Technical Details](#technical-details)

---

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- A functioning webcam
- Good lighting conditions
- OS: Windows, macOS, or Linux

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/face-gesture-2fa.git
cd face-gesture-2fa
```

### Step 2: Install Dependencies
```bash
pip install opencv-python face_recognition mediapipe numpy cryptography scikit-learn
```
### macOS
```bash
brew install cmake
pip install opencv-python face_recognition mediapipe numpy cryptography scikit-learn
```
### Linux
```bash
sudo apt-get install cmake python3-dev
pip install opencv-python face_recognition mediapipe numpy cryptography scikit-learn
```
### Step 3: Run the System
```bash
python 2FA-Recognition.py
```
## 🚀 Quick Start
### 1. Enroll a New User
- Choose Option 1 (Enroll New User)
- Enter your name
- Capture your face samples
- Perform your gesture
- Profile is encrypted and saved automatically

## 2. Verify Access
- Choose Option 2 (Verify User)
- Face the camera
- Perform your registered gesture
- ✅ Access granted if both verifications succeed

## 📖 Usage Guide
### Face Capture Tips
- Ensure good lighting and clear visibility
- Keep your face centered and steady
- Rotate slightly between captures
- Avoid backlighting or dark rooms
- Remove glasses for best accuracy

### Gesture Capture Tips
- Choose a unique, consistent gesture (✌️, 👍, 👌, etc.)
- Use the same hand each time
- Hold it steady during capture
- Keep your hand fully visible

## ⚙️ How It Works
### Face Recognition Algorithm
- Captures 15 high-quality face samples
- Computes 128D embeddings using deep learning
- Averages embeddings for stable representation
- Verifies using:
   - Euclidean Distance ≤ 0.40
   - Cosine Similarity ≥ 0.55
### Gesture Recognition Algorithm
- Detects 21 hand landmarks via MediaPipe
- Normalizes coordinates & scales for consistency
- Extracts 100+ geometric features:
    - Landmark coordinates
    - Angles, distances, orientation vectors
- Verifies gesture using similarity:
    - Best ≥ 0.90, Avg ≥ 0.88, ≥60% matches

## 🔬 Technical Details

### Encryption Pipeline
```sql
User Profile → Pickle Serialization → Fernet Encryption → AES-128 CBC → Binary Storage
```
### Face Pipeline
```objectivec
Camera → OpenCV → HOG Detection → CNN Encoding → Distance + Similarity → Decision
```
- Model: dlib ResNet
- Accuracy: 99.38% (LFW Benchmark)
- Encoding: 128D face embeddings

### Gesture Pipeline
```nginx
Camera → MediaPipe → 21 Landmarks → Feature Extraction → Similar
```
- Features: 100+
- Precision: Sub-pixel hand trackingity → Decision

## ⚠️ Disclaimer
This system is intended for research and educational use.
For production-grade deployment, implement:
- Liveness detection
- Professional security audit
- Biometric data compliance
- Secure key management (HSM)
- Regular security updates
