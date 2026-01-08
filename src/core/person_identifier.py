from deepface import DeepFace
import cv2
import os
import numpy as np

KNOWN_FACES_DIR = "dashboard_server/known_faces"
RECOGNITION_MODEL = "Facenet"

# Cache known face embeddings
known_embeddings = []

def load_known_faces():
    global known_embeddings
    known_embeddings = []

    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, file)
            name = os.path.splitext(file)[0]

            try:
                embedding = DeepFace.represent(img_path=path, model_name=RECOGNITION_MODEL, enforce_detection=False)[0]["embedding"]
                known_embeddings.append({"name": name, "embedding": np.array(embedding)})
                print(f"✅ Loaded known face: {name}")
            except Exception as e:
                print(f"⚠️ Failed to load {file}: {e}")

def identify_person(frame):
    """Identify a person in the frame. Returns name or 'Unknown'."""
    try:
        # Save temporary frame
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        # Extract embedding
        emb = DeepFace.represent(img_path=temp_path, model_name=RECOGNITION_MODEL, enforce_detection=False)[0]["embedding"]
        emb = np.array(emb)

        best_match = None
        best_similarity = -1

        for known in known_embeddings:
            sim = np.dot(emb, known["embedding"]) / (np.linalg.norm(emb) * np.linalg.norm(known["embedding"]))
            if sim > best_similarity:
                best_similarity = sim
                best_match = known["name"]

        os.remove(temp_path)

        if best_similarity > 0.75:
            return best_match
        else:
            return "Unknown"

    except Exception as e:
        print(f"⚠️ Recognition error: {e}")
        return "Unknown"

