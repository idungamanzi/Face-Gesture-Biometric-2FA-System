import os
import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import pickle
import base64
from cryptography.fernet import Fernet
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import time

FACE_THRESHOLD = 0.40  
GESTURE_THRESHOLD = 0.90  
FACE_SAMPLES = 15  
GESTURE_SESSIONS = 5  
GESTURE_SAMPLES_PER_SESSION = 3  
MIN_GESTURE_MATCHES = 0.88  
DATA_DIR = "data"
KEY_FILE = "secret.key"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def ensure_directories():
    # Create necessary directories if they don't exist.
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✓ Created {DATA_DIR} directory")

# Generate a new encryption key or load existing one.
def generate_or_load_key(key_file=KEY_FILE):
    if os.path.exists(key_file):
        with open(key_file, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
        print(f"✓ Generated new encryption key: {key_file}")
        return key

# Encrypt data using Fernet symmetric encryption.
def encrypt_data(data, key_file=KEY_FILE):
    key = generate_or_load_key(key_file)
    fernet = Fernet(key)
    serialized = pickle.dumps(data)
    encrypted = fernet.encrypt(serialized)
    return encrypted

#Decrypt data using Fernet symmetric encryption.
def decrypt_data(token, key_file=KEY_FILE):
    key = generate_or_load_key(key_file)
    fernet = Fernet(key)
    decrypted = fernet.decrypt(token)
    data = pickle.loads(decrypted)
    return data

#Get the next available user ID.
def get_next_user_id():
    if not os.path.exists(DATA_DIR):
        return 1
    existing_users = [d for d in os.listdir(DATA_DIR) if d.startswith("user_")]
    if not existing_users:
        return 1
    user_ids = [int(d.split("_")[1]) for d in existing_users]
    return max(user_ids) + 1


def calculate_face_quality(face_location, frame_shape):
    """
    Calculate face quality metrics to ensure good samples.
    Returns quality score (0-1, higher is better).
    """
    top, right, bottom, left = face_location
    face_width = right - left
    face_height = bottom - top
    frame_height, frame_width = frame_shape[:2]
    
    # Face should occupy 15-40% of frame width
    face_ratio = face_width / frame_width
    
    # Check if face is centered
    face_center_x = (left + right) / 2
    frame_center_x = frame_width / 2
    center_offset = abs(face_center_x - frame_center_x) / frame_width
    
    # Quality score based on size and centering
    size_score = 1.0 if 0.15 <= face_ratio <= 0.40 else 0.5
    center_score = 1.0 - center_offset
    
    quality = (size_score + center_score) / 2
    return quality


def capture_face_samples(name, uid):
    """
    Capture multiple high-quality face samples and compute average embedding.
    Uses stricter quality checks to ensure accurate face profiles.
    """
    print(f"\n Starting face capture for {name}...")
    print(f"Please look at the camera. Capturing {FACE_SAMPLES} high-quality samples.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Could not access webcam")
        return None
    
    # Set higher resolution for better face detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    encodings = []
    captured = 0
    
    print("\n IMPORTANT INSTRUCTIONS:")
    print("- Look directly at the camera")
    print("- Ensure your face is well-lit and clearly visible")
    print("- Face should be centered and occupy 15-40% of the frame")
    print("- Rotate your head slightly between captures (left, right, up, down)")
    print("- Remove glasses/accessories if possible for better accuracy")
    print("\nPress 'c' to capture or 'q' to quit\n")
    
    while captured < FACE_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame")
            break
        
        # Display frame with face detection
        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        # Draw rectangles around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Captured: {captured}/{FACE_SAMPLES}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'c' to capture", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(face_locations) == 1:
            quality = calculate_face_quality(face_locations[0], frame.shape)
            quality_text = f"Quality: {quality:.2f}"
            color = (0, 255, 0) if quality > 0.7 else (0, 165, 255)
            cv2.putText(display_frame, quality_text, 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Face Capture', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(face_locations) == 0:
                print(f"No face detected. Please ensure your face is visible.")
                continue
            elif len(face_locations) > 1:
                print(f"Multiple faces detected ({len(face_locations)}). Please ensure only YOUR face is visible.")
                continue
            
            # Check face quality
            quality = calculate_face_quality(face_locations[0], frame.shape)
            if quality < 0.6:
                print(f"Face quality too low ({quality:.2f}). Please center your face and move closer.")
                continue
            
            # Get face encoding with higher accuracy model
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations, num_jitters=10, model="large")
            
            if len(face_encodings) > 0:
                encodings.append(face_encodings[0])
                captured += 1
                print(f"✓ Sample {captured}/{FACE_SAMPLES} captured (quality: {quality:.2f})")
                time.sleep(0.5)
        
        elif key == ord('q'):
            print(" Face capture cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(encodings) < FACE_SAMPLES:
        print(f" Insufficient face samples captured ({len(encodings)}/{FACE_SAMPLES})")
        return None
    
    # Compute average embedding
    avg_encoding = np.mean(encodings, axis=0)
    
    # Calculate variance to ensure consistency
    variance = np.var(encodings, axis=0)
    avg_variance = np.mean(variance)
    
    print(f"✓ Face profile created from {len(encodings)} samples")
    print(f"  Profile consistency: {(1.0 - min(avg_variance, 1.0)):.2%}")
    
    return avg_encoding


def normalize_landmarks_advanced(landmarks):
    """
    Advanced normalization that preserves precise gesture characteristics.
    Captures exact hand configuration including finger positions and angles.
    """
    # Extract coordinates
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    # Center at wrist (landmark 0)
    centered = coords - coords[0]
    
    # Normalize by scale (distance from wrist to middle finger tip)
    middle_finger_tip = centered[12]  # Landmark 12 is middle finger tip
    scale = np.linalg.norm(middle_finger_tip)
    
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    
    # Calculate additional geometric features for accuracy
    features = []
    
    # 1. All normalized landmark coordinates
    features.extend(normalized.flatten())
    
    # 2. Angles between key finger segments
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    finger_bases = [2, 5, 9, 13, 17]  # Finger base joints
    
    for tip, base in zip(finger_tips, finger_bases):
        vector = normalized[tip] - normalized[base]
        features.extend(vector)
    
    # 3. Distances between finger tips (captures hand shape)
    for i in range(len(finger_tips)):
        for j in range(i + 1, len(finger_tips)):
            dist = np.linalg.norm(normalized[finger_tips[i]] - normalized[finger_tips[j]])
            features.append(dist)
    
    # 4. Palm orientation (normal vector)
    wrist = normalized[0]
    index_base = normalized[5]
    pinky_base = normalized[17]
    v1 = index_base - wrist
    v2 = pinky_base - wrist
    normal = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 0:
        normal = normal / normal_norm
    features.extend(normal)
    
    return np.array(features)


def capture_gesture_samples(name, uid):
    """
    Capture precise gesture samples across multiple sessions.
    Records exact hand configuration and movements for accurate verification.
    """
    print(f"\n✋ Starting gesture capture for {name}...")
    print(f"You will perform your gesture {GESTURE_SESSIONS} times.")
    print(f"Each session captures {GESTURE_SAMPLES_PER_SESSION} samples of your gesture.\n")
    
    print(" IMPORTANT GESTURE INSTRUCTIONS:")
    print("- Choose a UNIQUE gesture (specific hand position/movement)")
    print("- Use the SAME hand(s) every time")
    print("- Keep fingers in the EXACT same configuration")
    print("- If using movement, make it CONSISTENT")
    print("- The system will learn YOUR specific gesture pattern")
    print("- Examples: Peace sign, thumbs up, specific finger combination")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Could not access webcam")
        return None
    
    # Set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    all_gestures = []
    hand_labels = []  # Track which hand (left/right)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Allow detecting both hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        for session in range(GESTURE_SESSIONS):
            print(f"\n Session {session + 1}/{GESTURE_SESSIONS}")
            print("Instructions:")
            print("- Show your gesture clearly")
            print("- Hold the gesture STEADY")
            print("- Ensure good lighting")
            print("- Press 'g' when ready to capture")
            print("- Press 'q' to quit\n")
            
            session_gestures = []
            session_labels = []
            captured = 0
            
            while captured < GESTURE_SAMPLES_PER_SESSION:
                ret, frame = cap.read()
                if not ret:
                    print(" Failed to capture frame")
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = hands.process(rgb_frame)
                
                # Draw hand landmarks
                display_frame = frame.copy()
                hand_info = "No hands detected"
                
                if results.multi_hand_landmarks and results.multi_handedness:
                    num_hands = len(results.multi_hand_landmarks)
                    hand_types = [hand.classification[0].label for hand in results.multi_handedness]
                    hand_info = f"{num_hands} hand(s): {', '.join(hand_types)}"
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
                
                # Display info
                cv2.putText(display_frame, f"Session {session + 1}/{GESTURE_SESSIONS} - Captured: {captured}/{GESTURE_SAMPLES_PER_SESSION}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, hand_info, 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, "Press 'g' to capture gesture", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Gesture Capture', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('g'):
                    if results.multi_hand_landmarks and results.multi_handedness:
                        # Capture all visible hands
                        gesture_snapshot = {
                            'hands': [],
                            'num_hands': len(results.multi_hand_landmarks)
                        }
                        
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            hand_label = handedness.classification[0].label
                            normalized = normalize_landmarks_advanced(hand_landmarks)
                            
                            gesture_snapshot['hands'].append({
                                'label': hand_label,
                                'features': normalized
                            })
                        
                        session_gestures.append(gesture_snapshot)
                        captured += 1
                        hand_desc = f"{gesture_snapshot['num_hands']} hand(s)"
                        print(f"✓ Gesture sample {captured}/{GESTURE_SAMPLES_PER_SESSION} captured ({hand_desc})")
                        time.sleep(0.4)
                    else:
                        print("No hand detected. Please show your gesture clearly.")
                
                elif key == ord('q'):
                    print(" Gesture capture cancelled")
                    cap.release()
                    cv2.destroyAllWindows()
                    return None
            
            all_gestures.extend(session_gestures)
            print(f"✓ Session {session + 1} complete")
            
            if session < GESTURE_SESSIONS - 1:
                print("\nRelax for a moment before the next session...")
                time.sleep(2)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(all_gestures) == 0:
        print(" No gesture samples captured")
        return None
    
    # Analyze gesture consistency
    num_hands_list = [g['num_hands'] for g in all_gestures]
    expected_hands = max(set(num_hands_list), key=num_hands_list.count)
    
    print(f"\n✓ Gesture profile created from {len(all_gestures)} samples")
    print(f"  Detected hand configuration: {expected_hands} hand(s)")
    
    # Return structured gesture data
    gesture_data = {
        'samples': all_gestures,
        'expected_hands': expected_hands,
        'total_samples': len(all_gestures)
    }
    
    return gesture_data

"""
Strictly verify face against stored embedding.
Uses multiple comparison metrics for higher accuracy.
"""
def verify_face(known_embedding, test_frame):
    rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces with higher accuracy
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    
    if len(face_locations) == 0:
        return False, None, "No face detected"
    
    if len(face_locations) > 1:
        return False, None, f"Multiple faces detected ({len(face_locations)}). Only registered user's face should be visible"
    
    # Check face quality
    quality = calculate_face_quality(face_locations[0], test_frame.shape)
    if quality < 0.5:
        return False, None, f"Face quality too low ({quality:.2f}). Please position yourself better"
    
    # Get encoding with high accuracy
    face_encodings = face_recognition.face_encodings(
        rgb_frame, face_locations, num_jitters=10, model="large")
    
    if len(face_encodings) == 0:
        return False, None, "Could not encode face"
    
    test_encoding = face_encodings[0]
    
    # Calculate multiple distance metrics for robust verification
    euclidean_distance = np.linalg.norm(known_embedding - test_encoding)
    similarity = cosine_similarity([known_embedding], [test_encoding])[0][0]
    manhattan_distance = np.sum(np.abs(known_embedding - test_encoding))
    euclidean_pass = euclidean_distance <= FACE_THRESHOLD
    similarity_pass = similarity >= 0.55  # Minimum cosine similarity
    
    success = euclidean_pass and similarity_pass
    
    metrics = {
        'euclidean_distance': euclidean_distance,
        'cosine_similarity': similarity,
        'manhattan_distance': manhattan_distance,
        'quality': quality
    }
    
    message = f"Distance: {euclidean_distance:.4f} (≤{FACE_THRESHOLD}), " \
              f"Similarity: {similarity:.4f} (≥0.55), Quality: {quality:.2f}"
    
    if not euclidean_pass:
        message += " | FAIL: Face distance too high (not matching registered face)"
    elif not similarity_pass:
        message += " | FAIL: Similarity too low (not matching registered face)"
    
    return success, metrics, message

"""
Compare two gesture snapshots with strict accuracy.
Checks hand count, hand labels, and feature similarity.
"""
def compare_gestures(gesture1, gesture2):
    # Must have same number of hands
    if gesture1['num_hands'] != gesture2['num_hands']:
        return 0.0
    
    # If no hands, return 0
    if gesture1['num_hands'] == 0:
        return 0.0
    
    # Match hands by label
    similarities = []
    
    for hand1 in gesture1['hands']:
        best_match = 0.0
        for hand2 in gesture2['hands']:
            # Hands must be the same type (left/right)
            if hand1['label'] == hand2['label']:
                # Calculate cosine similarity
                sim = cosine_similarity([hand1['features']], [hand2['features']])[0][0]
                best_match = max(best_match, sim)
        
        similarities.append(best_match)
    
    # Return average similarity across all hands
    return np.mean(similarities) if similarities else 0.0


def verify_gesture(known_gesture_data, test_gesture):
    """
    Strictly verify gesture against all stored templates.
    All comparisons must meet threshold for verification to pass.
    
    Args:
        known_gesture_data: Dictionary containing gesture samples
        test_gesture: Current gesture snapshot
    
    Returns:
        (success, metrics_dict, message)
    """
    if test_gesture is None or known_gesture_data is None:
        return False, None, "Invalid gesture data"
    
    known_samples = known_gesture_data['samples']
    expected_hands = known_gesture_data['expected_hands']
    
    # Check if number of hands matches
    if test_gesture['num_hands'] != expected_hands:
        return False, None, f"Hand count mismatch: Expected {expected_hands}, got {test_gesture['num_hands']}"
    
    # Compare with all stored gestures
    similarities = []
    for known in known_samples:
        sim = compare_gestures(known, test_gesture)
        similarities.append(sim)
    
    if len(similarities) == 0:
        return False, None, "No gesture comparisons possible"
    
    # Calculate metrics
    best_similarity = max(similarities)
    avg_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)
    
    # Count how many samples meet the threshold
    matches_above_threshold = sum(1 for s in similarities if s >= GESTURE_THRESHOLD)
    match_percentage = matches_above_threshold / len(similarities)
    
    best_pass = best_similarity >= GESTURE_THRESHOLD
    avg_pass = avg_similarity >= MIN_GESTURE_MATCHES
    percentage_pass = match_percentage >= 0.60
    
    success = best_pass and avg_pass and percentage_pass
    
    metrics = {
        'best_similarity': best_similarity,
        'avg_similarity': avg_similarity,
        'median_similarity': median_similarity,
        'match_percentage': match_percentage,
        'matches': matches_above_threshold,
        'total': len(similarities)
    }
    
    message = f"Best: {best_similarity:.4f} (≥{GESTURE_THRESHOLD}), " \
              f"Avg: {avg_similarity:.4f} (≥{MIN_GESTURE_MATCHES}), " \
              f"Matches: {matches_above_threshold}/{len(similarities)} ({match_percentage:.1%})"
    
    if not best_pass:
        message += " | FAIL: Best match too low (gesture doesn't match registered gesture)"
    elif not avg_pass:
        message += " | FAIL: Average similarity too low (inconsistent with registered gesture)"
    elif not percentage_pass:
        message += " | FAIL: Too few samples match (not the registered gesture)"
    
    return success, metrics, message


def enroll_new_user():
    """Enroll a new user with face and gesture data."""
    print("\n" + "="*60)
    print(" NEW USER ENROLLMENT")
    print("="*60)
    
    ensure_directories()
    
    # Get user information
    name = input("\nEnter user name: ").strip()
    if not name:
        print(" Name cannot be empty")
        return
    
    uid = get_next_user_id()
    user_dir = os.path.join(DATA_DIR, f"user_{uid}")
    os.makedirs(user_dir, exist_ok=True)
    
    print(f"\n Assigned User ID: {uid}")
    
    # Capture face samples
    face_embedding = capture_face_samples(name, uid)
    if face_embedding is None:
        print(" Face enrollment failed")
        return
    
    # Capture gesture samples
    gesture_data = capture_gesture_samples(name, uid)
    if gesture_data is None:
        print(" Gesture enrollment failed")
        return
    
    # Create user profile
    profile = {
        'uid': uid,
        'name': name,
        'face_embedding': face_embedding,
        'gesture_data': gesture_data,
        'enrolled_at': datetime.now().isoformat()
    }
    
    # Encrypt and save
    profile_path = os.path.join(user_dir, "profile.bin")
    encrypted = encrypt_data(profile)
    
    with open(profile_path, 'wb') as f:
        f.write(encrypted)
    
    print(f"\n Enrollment successful!")
    print(f"   User: {name}")
    print(f"   ID: {uid}")
    print(f"   Profile: {profile_path}")
    print(f"   Face samples: {FACE_SAMPLES}")
    print(f"   Gesture samples: {gesture_data['total_samples']}")
    print(f"   Gesture type: {gesture_data['expected_hands']} hand(s)")


def load_all_users():
    """Load all enrolled user profiles."""
    if not os.path.exists(DATA_DIR):
        return []
    
    users = []
    user_dirs = [d for d in os.listdir(DATA_DIR) if d.startswith("user_")]
    
    for user_dir in user_dirs:
        profile_path = os.path.join(DATA_DIR, user_dir, "profile.bin")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'rb') as f:
                    encrypted = f.read()
                profile = decrypt_data(encrypted)
                users.append(profile)
            except Exception as e:
                print(f"Warning: Could not load {user_dir}: {e}")
    
    return users


def verify_user():
    """Verify a user using strict face and gesture recognition."""
    print("\n" + "="*60)
    print(" USER VERIFICATION")
    print("="*60)
    
    # Load all users
    users = load_all_users()
    
    if len(users) == 0:
        print("\n No enrolled users found. Please enroll first.")
        return
    
    print(f"\n Found {len(users)} enrolled user(s)")
    
    # Phase 1: Face Verification
    print("\n" + "-"*60)
    print("PHASE 1: FACE VERIFICATION")
    print("-"*60)
    print("\️  IMPORTANT:")
    print("- Only the registered user's face should be visible")
    print("- Ensure good lighting and clear visibility")
    print("- Face the camera directly")
    print("\nPress 'v' to verify | Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Could not access webcam")
        return
    
    # Set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    face_verified = False
    matched_user = None
    
    while not face_verified:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame")
            break
        
        display_frame = frame.copy()
        
        # Show face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 255), 2)
        
        cv2.putText(display_frame, "FACE VERIFICATION - Press 'v' to verify", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Faces detected: {len(face_locations)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Verification', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('v'):
            print("\n Verifying face...")
            
            # Try to match with each user
            best_match = None
            best_distance = float('inf')
            best_metrics = None
            
            for user in users:
                success, metrics, message = verify_face(user['face_embedding'], frame)
                
                print(f"\nChecking against: {user['name']}")
                print(f"  {message}")
                
                if success and metrics and metrics['euclidean_distance'] < best_distance:
                    best_distance = metrics['euclidean_distance']
                    best_match = user
                    best_metrics = metrics
            
            if best_match:
                matched_user = best_match
                print(f"\n Face verified: {matched_user['name']}")
                print(f"   Distance: {best_metrics['euclidean_distance']:.4f}")
                print(f"   Similarity: {best_metrics['cosine_similarity']:.4f}")
                print(f"   Quality: {best_metrics['quality']:.2f}")
                face_verified = True
            else:
                print("\n Face verification FAILED")
                print("   Your face does not match any registered user")
                print("   Reasons could be:")
                print("   - You are not enrolled in the system")
                print("   - Poor lighting or face angle")
                print("   - Multiple faces detected")
                print("\nTry again or press 'q' to quit")
        
        elif key == ord('q'):
            print(" Verification cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    if not face_verified:
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Phase 2: Gesture Verification
    print("\n" + "-"*60)
    print("PHASE 2: GESTURE VERIFICATION")
    print("-"*60)
    print(f"\n Verifying gesture for: {matched_user['name']}")
    print(f"   Expected: {matched_user['gesture_data']['expected_hands']} hand(s)")
    
    print("\  IMPORTANT:")
    print("- Perform the EXACT gesture you registered")
    print("- Use the SAME hand(s)")
    print("- Keep finger positions IDENTICAL")
    print("- Hold the gesture steady")
    print("\nPress 'g' to verify | Press 'q' to quit")
    
    gesture_verified = False
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        while not gesture_verified:
            ret, frame = cap.read()
            if not ret:
                print(" Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            display_frame = frame.copy()
            hand_info = "No hands detected"
            num_hands = 0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                num_hands = len(results.multi_hand_landmarks)
                hand_types = [hand.classification[0].label for hand in results.multi_handedness]
                hand_info = f"{num_hands} hand(s): {', '.join(hand_types)}"
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
            
            cv2.putText(display_frame, "GESTURE VERIFICATION - Press 'g' to verify", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, hand_info, 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Expected: {matched_user['gesture_data']['expected_hands']} hand(s)", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Verification', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('g'):
                if results.multi_hand_landmarks and results.multi_handedness:
                    print("\n Verifying gesture...")
                    
                    # Capture current gesture
                    test_gesture = {
                        'hands': [],
                        'num_hands': len(results.multi_hand_landmarks)
                    }
                    
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        normalized = normalize_landmarks_advanced(hand_landmarks)
                        
                        test_gesture['hands'].append({
                            'label': hand_label,
                            'features': normalized
                        })
                    
                    # Verify gesture
                    success, metrics, message = verify_gesture(
                        matched_user['gesture_data'], test_gesture)
                    
                    print(f"  {message}")
                    
                    if success:
                        print(f"\n Gesture verified")
                        print(f"   Best match: {metrics['best_similarity']:.4f}")
                        print(f"   Average match: {metrics['avg_similarity']:.4f}")
                        print(f"   Samples matched: {metrics['matches']}/{metrics['total']}")
                        gesture_verified = True
                    else:
                        print("\n Gesture verification FAILED")
                        print("   Your gesture does not match the registered gesture")
                        print("   Reasons could be:")
                        print("   - Wrong hand (left vs right)")
                        print("   - Different number of hands")
                        print("   - Finger positions not matching")
                        print("   - Hand orientation different")
                        print("\nTry again or press 'q' to quit")
                else:
                    print(" No hand detected. Please show your gesture clearly.")
            
            elif key == ord('q'):
                print(" Verification cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final decision
    print("\n" + "="*60)
    if face_verified and gesture_verified:
        print(" ACCESS GRANTED")
        print(f"   Welcome, {matched_user['name']}!")
        print(f"   User ID: {matched_user['uid']}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Authentication: Face + Gesture ✓")
    else:
        print(" ACCESS DENIED")
        print("   Both face and gesture verification must pass")
        if face_verified:
            print("   ✓ Face verified")
            print("   ✗ Gesture failed")
        else:
            print("   ✗ Face failed")
    print("="*60)


def main_menu():
    print("\n" + "="*60)
    print(" FACE + GESTURE 2FA ACCESS CONTROL SYSTEM")
    print("   High-Accuracy Biometric Authentication")
    print("="*60)
    
    while True:
        print("\n MAIN MENU")
        print("-" * 40)
        print("1.  Enroll New User")
        print("2.  Verify User")
        print("3.  Exit")
        print("-" * 40)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            enroll_new_user()
        elif choice == '2':
            verify_user()
        elif choice == '3':
            print("\n Goodbye!")
            print("System secured. All profiles encrypted.")
            break
        else:
            print(" Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n Program interrupted by user")
    except Exception as e:
        print(f"\n An error occurred: {e}")
        import traceback
        traceback.print_exc()