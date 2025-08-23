import argparse
import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import os
import sqlite3
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from cryptography.fernet import Fernet
from datetime import datetime

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = APP_DIR / "users.db"
KEY_PATH = APP_DIR / "secret.key"

# Face recognition configurations
FACE_TOLERANCE = 0.52   
FRAME_DOWNSCALE = 0.25 
MAX_FACE_SAMPLES = 12  
# Gesture configurations
GESTURE_SECONDS = 3.0   
GESTURE_FPS = 15       
GESTURE_SAMPLES = 5    
DTW_SAKOE = 8           

# Drawing/visuals
FONT = cv2.FONT_HERSHEY_SIMPLEX

#Crypto helpers
def get_fernet() -> Fernet:
    if not KEY_PATH.exists():
        key = Fernet.generate_key()
        KEY_PATH.write_bytes(key)
        print(f"[INFO] Generated encryption key at {KEY_PATH}")
    else:
        key = KEY_PATH.read_bytes()
    return Fernet(key)

fernet = get_fernet()

def encrypt_bytes(b: bytes) -> bytes:
    return fernet.encrypt(b)

def decrypt_bytes(b: bytes) -> bytes:
    return fernet.decrypt(b)

# DB helpers
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS blobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        type TEXT NOT NULL,         -- 'faces' or 'gestures' or 'meta'
        path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)
    con.commit()
    con.close()

def db_connect():
    return sqlite3.connect(DB_PATH)

def get_users() -> List[Tuple[int,str,str]]:
    con = db_connect(); cur = con.cursor()
    cur.execute("SELECT id, name, created_at FROM users ORDER BY id ASC")
    rows = cur.fetchall()
    con.close()
    return rows

def add_user(name: str) -> int:
    con = db_connect(); cur = con.cursor()
    cur.execute("INSERT INTO users (name, created_at) VALUES (?, ?)", (name, datetime.now().isoformat()))
    uid = cur.lastrowid
    con.commit(); con.close()
    return uid

def delete_user(user_id: int):
    con = db_connect(); cur = con.cursor()
    # delete blob files first
    cur.execute("SELECT path FROM blobs WHERE user_id=?", (user_id,))
    for (p,) in cur.fetchall():
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    # delete metadata folders
    user_dir = DATA_DIR / f"user_{user_id}"
    if user_dir.exists():
        for root, dirs, files in os.walk(user_dir, topdown=False):
            for name in files: 
                Path(root, name).unlink(missing_ok=True)
            for name in dirs:
                Path(root, name).rmdir()
        user_dir.rmdir()
    # delete DB rows
    cur.execute("DELETE FROM blobs WHERE user_id=?", (user_id,))
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    con.commit(); con.close()

def add_blob(user_id: int, btype: str, path: Path):
    con = db_connect(); cur = con.cursor()
    cur.execute("INSERT INTO blobs (user_id, type, path, created_at) VALUES (?, ?, ?, ?)",
                (user_id, btype, str(path), datetime.now().isoformat()))
    con.commit(); con.close()

def get_user_blobs(user_id: int, btype: str) -> List[str]:
    con = db_connect(); cur = con.cursor()
    cur.execute("SELECT path FROM blobs WHERE user_id=? AND type=?", (user_id, btype))
    rows = [r[0] for r in cur.fetchall()]
    con.close()
    return rows

# File helpers
def ensure_dirs(user_id: int):
    (DATA_DIR / f"user_{user_id}" / "faces").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / f"user_{user_id}" / "gestures").mkdir(parents=True, exist_ok=True)

def save_encrypted_json(user_id: int, btype: str, obj: Dict[str, Any]) -> Path:
    payload = json.dumps(obj).encode("utf-8")
    enc = encrypt_bytes(payload)
    out = DATA_DIR / f"user_{user_id}" / f"{btype}_{int(time.time())}.bin"
    out.write_bytes(enc)
    add_blob(user_id, btype, out)
    return out

# DTW (fast-ish with Sakoe-Chiba band)
def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band: int = DTW_SAKOE) -> float:
    Ta, Tb = len(seq_a), len(seq_b)
    D = np.full((Ta+1, Tb+1), np.inf, dtype=np.float32)
    D[0,0] = 0.0
    for i in range(1, Ta+1):
        j_start = max(1, i - band)
        j_end = min(Tb, i + band)
        ai = seq_a[i-1]
        for j in range(j_start, j_end+1):
            bj = seq_b[j-1]
            cost = np.linalg.norm(ai - bj)
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[Ta, Tb])

def normalize_landmarks(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((1, 1))

    expected_dim = 468*3 + 21*3 + 21*3

    normed = []
    for f in frames:
        f = f.flatten()
        if f.shape[0] < expected_dim:
            f = np.pad(f, (0, expected_dim - f.shape[0]), constant_values=0)
        elif f.shape[0] > expected_dim:
            f = f[:expected_dim]
        normed.append(f)

    # normalize relative to face center
    arr = np.stack(normed, axis=0).astype(np.float32)
    arr -= np.mean(arr, axis=0, keepdims=True)
    arr /= (np.std(arr, axis=0, keepdims=True) + 1e-6)
    return arr

    flat = []
    for f in seq:
        arr = np.asarray(f, dtype=np.float32)
        # use only x,y if z exists
        if arr.shape[1] >= 2:
            arr2d = arr[:, :2]
        else:
            arr2d = arr
        # center & scale per frame
        mean = arr2d.mean(axis=0, keepdims=True)
        centered = arr2d - mean
        norm = np.linalg.norm(centered) + 1e-6
        scaled = centered / norm
        flat.append(scaled.flatten())
    return np.stack(flat, axis=0)  # [T, D]

# MediaPipe pipelines
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_face_landmarks(image_bgr, face_landmarks):
    if face_landmarks:
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())

def extract_hand_landmarks(results) -> np.ndarray | None:
    hand = results.right_hand_landmarks or results.left_hand_landmarks
    if hand is None:
        return None
    pts = []
    for lm in hand.landmark:
        pts.append([lm.x, lm.y, lm.z])
    return np.array(pts, dtype=np.float32)

# Face enrollment & recognition
def capture_face_embeddings(name: str, user_id: int) -> Dict[str, Any]:
    print("[ENROLL] Face capture starting... Look at the camera. Press 'q' to abort.")
    cap = cv2.VideoCapture(0)
    embeddings = []
    raw_frames = []
    alert_shown = False
    try:
        while len(embeddings) < MAX_FACE_SAMPLES:
            ok, frame = cap.read()
            if not ok: 
                print("[WARN] Camera read failed."); break
            overlay = frame.copy()
            h, w = frame.shape[:2]
            for i in range(0, w, 40):
                cv2.line(overlay, (i,0), (i,h), (255,255,255), 1)
            for j in range(0, h, 40):
                cv2.line(overlay, (0,j), (w,j), (255,255,255), 1)
            alpha = 0.12
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
            
            small = cv2.resize(frame, (0,0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_small, model='hog')
            encs = face_recognition.face_encodings(rgb_small, boxes)
            
            for (top, right, bottom, left) in boxes:
                # rescale back
                top = int(top / FRAME_DOWNSCALE); bottom = int(bottom / FRAME_DOWNSCALE)
                left = int(left / FRAME_DOWNSCALE); right = int(right / FRAME_DOWNSCALE)
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, f"Capturing faces: {len(embeddings)}/{MAX_FACE_SAMPLES}", (10,30), FONT, 0.8, (0,255,0), 2)
            cv2.imshow("Enrollment - Face", frame)
            
            if encs:
                embeddings.append(encs[0].tolist())
                raw_frames.append(cv2.imencode(".jpg", frame)[1].tobytes())
                if not alert_shown and len(embeddings) >= MAX_FACE_SAMPLES//2:
                    print("[ALERT] Face capture halfway. Keep steady...")
                    alert_shown = True
                time.sleep(0.05)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Aborted by user.")
                break
    finally:
        cap.release(); cv2.destroyAllWindows()
    print("[ENROLL] Face capture complete." if len(embeddings) >= MAX_FACE_SAMPLES else "[ENROLL] Face capture incomplete.")
   
    ensure_dirs(user_id)
    out = save_encrypted_json(user_id, "faces", {
        "name": name,
        "embeddings": embeddings,
        "raw_frames_jpg": [list(b) for b in raw_frames],  # store as list of ints for json
        "face_tolerance": FACE_TOLERANCE,
    })
    print(f"[SAVE] Encrypted face blob -> {out}")
    return {"embeddings": embeddings, "blob_path": str(out)}

def recognize_user_from_frame(frame_bgr, all_users_data) -> Tuple[int | None, float | None, Tuple[int,int,int,int] | None]:
    """
    all_users_data: list of (user_id, name, embeddings: np.ndarray[K,128])
    Returns (user_id, min_distance, bbox)
    """
    small = cv2.resize(frame_bgr, (0,0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_small, model='hog')
    if not boxes:
        return None, None, None
    encs = face_recognition.face_encodings(rgb_small, boxes)
    if not encs:
        return None, None, None
    enc = encs[0]
    
    (top, right, bottom, left) = boxes[0]
    top = int(top / FRAME_DOWNSCALE); bottom = int(bottom / FRAME_DOWNSCALE)
    left = int(left / FRAME_DOWNSCALE); right = int(right / FRAME_DOWNSCALE)
   
    best_uid = None
    best_dist = 999
    for (uid, name, emb_array) in all_users_data:
        dists = face_recognition.face_distance(emb_array, enc)
        m = float(np.min(dists)) if len(dists)>0 else 999
        if m < best_dist:
            best_dist = m; best_uid = uid
    return best_uid, best_dist, (left, top, right, bottom)

# Gesture enrollment & verification 
def capture_gesture_samples(name: str, user_id: int) -> Dict[str, Any]:
    print("[ENROLL] Gesture capture starting... Perform your personal gesture when prompted.")
    print("         Each sample ~{:.1f}s. Press 'q' to abort.".format(GESTURE_SECONDS))
    cap = cv2.VideoCapture(0)
    samples = []
    raw_sequences = []  # each is list of JPEG bytes per frame
    
    with mp_holistic.Holistic(
        model_complexity=1, refine_face_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        try:
            for s in range(GESTURE_SAMPLES):
                print(f"[ALERT] Starting gesture sample {s+1}/{GESTURE_SAMPLES} in 2s...")
                start_wait = time.time()
                while time.time() - start_wait < 2.0:
                    ok, frame = cap.read()
                    if not ok: break
                    cv2.putText(frame, "Get ready...", (10,30), FONT, 1, (0,255,255), 2)
                    cv2.imshow("Enrollment - Gesture", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        raise KeyboardInterrupt
                # record window
                seq = []
                raw_frames = []
                frames_needed = int(GESTURE_SECONDS * GESTURE_FPS)
                last = time.time()
                while len(seq) < frames_needed:
                    ok, frame = cap.read()
                    if not ok: 
                        print("[WARN] Camera read failed."); break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)
                
                    hand_pts = extract_hand_landmarks(results)
                    if hand_pts is None and results.face_landmarks:
                        # take subset of face landmarks (contours) to keep D small
                        pts = []
                        for lm in results.face_landmarks.landmark[0:100]:
                            pts.append([lm.x, lm.y, lm.z])
                        hand_pts = np.array(pts, dtype=np.float32)
                    if hand_pts is not None:
                        seq.append(hand_pts)
                    # overlays
                    if results.face_landmarks:
                        draw_face_landmarks(frame, results.face_landmarks)
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    cv2.putText(frame, f"Recording gesture {s+1}/{GESTURE_SAMPLES}", (10,30), FONT, 0.8, (0,255,0), 2)
                    cv2.imshow("Enrollment - Gesture", frame)
                    raw_frames.append(cv2.imencode(".jpg", frame)[1].tobytes())
                    # pace capture
                    now = time.time()
                    delay = max(0.0, (1.0/GESTURE_FPS) - (now - last))
                    if delay > 0: time.sleep(delay)
                    last = time.time()
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        raise KeyboardInterrupt
                if seq:
                    samples.append([x.tolist() for x in seq])
                    raw_sequences.append([list(b) for b in raw_frames])
                print(f"[ALERT] Gesture sample {s+1} captured.")
        except KeyboardInterrupt:
            print("[INFO] Aborted by user.")
        finally:
            cap.release(); cv2.destroyAllWindows()
    print("[ENROLL] Gesture capture complete." if len(samples) >= 1 else "[ENROLL] Gesture capture incomplete.")
    # Compute per-user threshold (mean + 2*std of pairwise DTW between samples)
    normed = [normalize_landmarks([np.array(f) for f in sample]) for sample in samples]
    dists = []
    for i in range(len(normed)):
        for j in range(i+1, len(normed)):
            dists.append(dtw_distance(normed[i], normed[j]))
    if dists:
        th = float(np.mean(dists) + 2*np.std(dists))
    else:
        th = 12.0 
    ensure_dirs(user_id)
    out = save_encrypted_json(user_id, "gestures", {
        "name": name,
        "samples": samples,
        "raw_sequences_jpg": raw_sequences,
        "dtw_threshold": th,
        "sakoe_band": DTW_SAKOE,
        "fps": GESTURE_FPS,
        "seconds": GESTURE_SECONDS
    })
    print(f"[SAVE] Encrypted gesture blob -> {out}")
    print(f"[INFO] Learned DTW threshold ~ {th:.3f}")
    return {"samples": samples, "threshold": th, "blob_path": str(out)}

# Load all enrolled data
def load_all_users_data() -> List[Tuple[int,str,np.ndarray,Dict[str,Any]]]:
    """
    Returns list of (user_id, name, face_embeddings[K,128], gesture_meta_dict)
    gesture_meta_dict contains 'samples' (list) and 'dtw_threshold'
    """
    rows = get_users()
    all_data = []
    for uid, name, created_at in rows:
        # latest face blob
        face_paths = sorted(get_user_blobs(uid, "faces"))
        gest_paths = sorted(get_user_blobs(uid, "gestures"))
        if not face_paths or not gest_paths:
            continue
        face_data = json.loads(decrypt_bytes(Path(face_paths[-1]).read_bytes()).decode("utf-8"))
        gest_data = json.loads(decrypt_bytes(Path(gest_paths[-1]).read_bytes()).decode("utf-8"))
        emb = np.array(face_data.get("embeddings", []), dtype=np.float32)
        all_data.append((uid, name, emb, gest_data))
    return all_data

# Verification pipeline
def verify_auto():
    all_users = load_all_users_data()
    if not all_users:
        print("[ERROR] No fully enrolled users found (need both faces and gestures).")
        return
    print("[VERIFY] Starting verification. Step 1: Face → Step 2: Gesture")
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    try:
        recognized_uid = None
        recognized_name = None
        step = "face"
        gesture_result_text = ""
        while True:
            ok, frame = cap.read()
            if not ok: break
            display = frame.copy()
            # real-time face landmarks for feedback
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_res = face_mesh.process(rgb)
            if face_res.multi_face_landmarks:
                draw_face_landmarks(display, face_res.multi_face_landmarks[0])
            if step == "face":
                uid, dist, bbox = recognize_user_from_frame(frame, [(u, n, e) for (u,n,e,_) in all_users])
                if uid is not None and dist is not None and bbox is not None:
                    (l,t,r,b) = bbox
                    color = (0,255,0) if dist <= FACE_TOLERANCE else (0,0,255)
                    cv2.rectangle(display, (l,t), (r,b), color, 2)
                    cv2.putText(display, f"Face dist: {dist:.3f}", (l, t-10), FONT, 0.7, color, 2)
                    if dist <= FACE_TOLERANCE:
                        recognized_uid = uid
                        recognized_name = next(n for (u,n,_,_) in all_users if u==uid)
                        cv2.putText(display, f"Face recognized: {recognized_name}", (10,30), FONT, 0.9, (0,255,0), 2)
                        # proceed to gesture
                        step = "gesture"
                        print("[ALERT] Face recognized. Proceed to gesture phase.")
                        # brief pause to signal transition
                        t0 = time.time()
                        while time.time() - t0 < 1.2:
                            cv2.putText(display, "Prepare gesture...", (10,65), FONT, 0.9, (0,255,255), 2)
                            cv2.imshow("Verify", display)
                            cv2.waitKey(1)
                        continue
                else:
                    cv2.putText(display, "Face not recognized. Please show your face.", (10,30), FONT, 0.8, (0,0,255), 2)
            elif step == "gesture":
                # capture one gesture sequence and compare
                target = next(g for (u,_,_,g) in all_users if u==recognized_uid)
                th = float(target.get("dtw_threshold", 12.0))
                fps = int(target.get("fps", GESTURE_FPS))
                seconds = float(target.get("seconds", GESTURE_SECONDS))
                
                with mp_holistic.Holistic(
                    model_complexity=1, refine_face_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    frames_needed = int(seconds * fps)
                    seq = []
                    last = time.time()
                    for k in range(frames_needed):
                        ok, frame = cap.read()
                        if not ok: break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(rgb)
                        lm = extract_hand_landmarks(results)
                        if lm is None and results.face_landmarks:
                            pts = []
                            for lm in results.face_landmarks.landmark[0:100]:
                                pts.append([lm.x, lm.y, lm.z])
                            lm = np.array(pts, dtype=np.float32)
                        if lm is not None:
                            seq.append(lm)
                        # overlays
                        disp = frame.copy()
                        if results.face_landmarks:
                            draw_face_landmarks(disp, results.face_landmarks)
                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(disp, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(disp, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        cv2.putText(disp, "Recording gesture...", (10,30), FONT, 0.8, (0,255,0), 2)
                        cv2.imshow("Verify", disp)
                        # pacing
                        now = time.time()
                        delay = max(0.0, (1.0/fps) - (now - last))
                        if delay > 0: time.sleep(delay)
                        last = time.time()
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            break
                # compare
                if seq:
                    live = normalize_landmarks([np.array(f) for f in seq])
                    enroll_samples = target.get("samples", [])
                    dists = []
                    for s in enroll_samples:
                        norm = normalize_landmarks([np.array(f) for f in s])
                        dists.append(dtw_distance(live, norm))
                    best = min(dists) if dists else 999.0
                    print(f"[VERIFY] Gesture DTW distance: {best:.3f} (threshold {th:.3f})")
                    if best <= th:
                        gesture_result_text = f"Access granted. Welcome {recognized_name}"
                        color = (0,255,0)
                    else:
                        gesture_result_text = "Gesture did not match"
                        color = (0,0,255)
                    # show result for a moment
                    t0 = time.time()
                    while time.time() - t0 < 2.0:
                        disp = frame.copy()
                        cv2.putText(disp, gesture_result_text, (10,30), FONT, 0.9, color, 2)
                        cv2.imshow("Verify", disp)
                        cv2.waitKey(1)
                # reset pipeline to face step
                recognized_uid = None
                step = "face"
            
            cv2.imshow("Verify", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release(); face_mesh.close(); cv2.destroyAllWindows()

# Enrollment pipeline 
def enroll_new_user():
    name = input("Enter username to enroll: ").strip()
    if not name:
        print("[ERROR] username required.")
        return
    # create DB row
    try:
        uid = add_user(name)
    except sqlite3.IntegrityError:
        print("[ERROR] Name already exists. Choose another.")
        return
    ensure_dirs(uid)
    print(f"[INFO] Enrolling user '{name}' (id={uid})")
    # face phase
    face_info = capture_face_embeddings(name, uid)
    print("[ALERT] Face capture complete. Starting gesture capture...")
    # gesture phase
    gest_info = capture_gesture_samples(name, uid)
    print("[ENROLL] Enrollment finished for", name)

# List & Delete
def list_users_cli():
    rows = get_users()
    if not rows:
        print("(no users)")
        return
    print("ID\tName\tCreated")
    for (uid, name, created_at) in rows:
        print(f"{uid}\t{name}\t{created_at}")

def delete_user_cli():
    try:
        user_id = int(input("Enter user ID to delete: ").strip())
    except ValueError:
        print("[ERROR] Invalid ID."); return
    confirm = input(f"Type 'yes' to confirm deletion of user {user_id}: ").strip().lower()
    if confirm == 'yes':
        delete_user(user_id)
        print("[INFO] User deleted.")
    else:
        print("[INFO] Deletion cancelled.")

# Menu
def main_menu():
    init_db()
    while True:
        print("\n Face + Gesture 2FA ")
        print("1) Enroll new user")
        print("2) Verify (auto)")
        print("3) List users")
        print("4) Delete user")
        print("5) Quit")
        choice = input("Select an option: ").strip()
        if choice == '1':
            enroll_new_user()
        elif choice == '2':
            verify_auto()
        elif choice == '3':
            list_users_cli()
        elif choice == '4':
            delete_user_cli()
        elif choice == '5':
            print("Bye."); break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Exiting.")
