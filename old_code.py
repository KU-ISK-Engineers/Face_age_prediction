import cv2
import threading
import queue
from deepface import DeepFace
import numpy as np
from pypylon import pylon


# Constants for frame size and resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP = 5  # Skip every 5th frame for predictions (you can adjust this)

# Paths for the quantized models
frame_queue = queue.Queue(maxsize=2)
age_queue = queue.Queue(maxsize=2)
gender_queue = queue.Queue(maxsize=2)

# Locks to prevent race conditions
frame_lock = threading.Lock()
prediction_lock = threading.Lock()

# Initialize Video Capture with frame size settings

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Set auto exposure mode to continuous
camera.ExposureAuto.SetValue('Continuous')
# Set camera parameters
camera.AcquisitionMode.SetValue("Continuous")
        
        
# Set pixel format to RGB
camera.PixelFormat.SetValue("RGB8")
        
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Variables to store predictions
last_predictions = {}  # This will store predictions for each tracked bounding box (ID)

# Exit flag for all threads
exit_flag = threading.Event()

# Queue to handle frames with tracking data
def track_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_id_counter = 0  # To assign unique IDs to faces
    face_id_map = {}  # Maps face ID to bounding box

    while not exit_flag.is_set():
        image1 = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if image1.GrabSucceeded():
            # Convert the grabbed frame to OpenCV format
            frame = image1.Array
            
            frame = cv2.resize(frame,(FRAME_WIDTH, FRAME_HEIGHT))
                
            # Convert the RGB image to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Lock the frame for safe access
        with frame_lock:
            if faces is not None:
                # Store the frame with detected faces and their IDs
                for (x, y, w, h) in faces:
                    # Find matching ID for previously detected face based on position and size
                    tracked_id = None
                    for face_id, (bx, by, bw, bh) in face_id_map.items():
                        if abs(bx - x) < 10 and abs(by - y) < 10:  # Ensure face moved slightly or is same
                            tracked_id = face_id
                            break
                    
                    if tracked_id is None:  # New face detected
                        tracked_id = face_id_counter
                        face_id_counter += 1
                    
                    face_id_map[tracked_id] = (x, y, w, h)

                # Store frames with detected faces and their IDs
                frame_queue.put((frame, faces, face_id_map))

# Age and gender prediction thread
def predict_age_and_gender():
    global last_predictions
    frame_count = 0  # Used to track frame skipping
    while not exit_flag.is_set():
        frame, faces, face_id_map = frame_queue.get()
        if faces is None:
            continue

        # Make prediction for each bounding box in the current frame
        with prediction_lock:
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                tracked_id = None
                
                # Find corresponding tracked face ID
                for face_id, (bx, by, bw, bh) in face_id_map.items():
                    if (bx, by, bw, bh) == (x, y, w, h):  # Same face found, get ID
                        tracked_id = face_id
                        break

                if tracked_id is None:
                    continue

                # Skip prediction based on frame skipping logic
                if frame_count % FRAME_SKIP == 0:
                    try:
                        result = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)

                        # Process each result (for now, assuming a single face)
                        for res in result:
                            # Get the most confident gender and its confidence percentage
                            gender = max(res['gender'].items(), key=lambda x: x[1])[0]
                            gender_confidence = res['gender'][gender]
                            age = res.get('age', None)

                            # Store the prediction for this tracked face ID
                            last_predictions[tracked_id] = {
                                'age': age,
                                'gender': gender,
                                'gender_confidence': gender_confidence
                            }

                            # Debugging print statements to ensure data is being stored
                            print(f"Predicted for ID {tracked_id}: Age: {age}, Gender: {gender}, Gender Confidence: {gender_confidence}")
                    except Exception as e:
                        print(f"Error in DeepFace analysis: {e}")
                        continue
                
                frame_count += 1  # Increment frame count

# Display thread
def display_output():
    while not exit_flag.is_set():
        # Lock the frame for safe access
        with frame_lock:
            if not frame_queue.empty():
                frame, faces, face_id_map = frame_queue.get()
                
                for (x, y, w, h) in faces:
                    tracked_id = None

                    # Find corresponding tracked face ID
                    for face_id, (bx, by, bw, bh) in face_id_map.items():
                        if (bx, by, bw, bh) == (x, y, w, h):  # Same face found, get ID
                            tracked_id = face_id
                            break

                    if tracked_id is None:
                        continue

                    # Draw the bounding box around the face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display age
                    if tracked_id in last_predictions:
                        age = last_predictions[tracked_id].get('age', 'Unknown')
                        age_text = f"Age: {age}"
                        cv2.putText(frame, age_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

                    # Display gender with confidence
                    if tracked_id in last_predictions:
                        gender = last_predictions[tracked_id].get('gender', 'Unknown')
                        confidence = round(last_predictions[tracked_id].get('gender_confidence', 0), 2)
                        gender_text = f"Gender: {gender.capitalize()} ({confidence}%)"
                        cv2.putText(frame, gender_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

                # Display the frame
                cv2.imshow("Video Feed", frame)

        # Wait for key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            exit_flag.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# Create threads
tracking_thread = threading.Thread(target=track_faces, daemon=True)
age_and_gender_thread = threading.Thread(target=predict_age_and_gender, daemon=True)
display_thread = threading.Thread(target=display_output, daemon=True)

# Start threads
tracking_thread.start()
age_and_gender_thread.start()
display_thread.start()

# Wait for threads to complete (though they are daemon threads)
tracking_thread.join()
age_and_gender_thread.join()
display_thread.join()

# Signal exit for all threads
exit_flag.set()
