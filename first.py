import cv2
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from deepface import DeepFace
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
import numpy as np
import os # Import os module to remove the image file after sending

# Load YOLOv8 models
model = YOLO('yolov8n.pt')           # Object detection
pose_model = YOLO('yolov8n-pose.pt')     # Pose estimation
seg_model = YOLO('yolov8n-seg.pt')   # Segmentation model for person silhouettes


# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0)

# Email alert configuration
ALERT_COOLDOWN = 15   # seconds - Set this higher for testing, e.g., 5 seconds, then back to 60
last_alert_time = 0

EMAIL_SENDER = "aisurveillancesystem@gmail.com"
EMAIL_PASSWORD = "exvp kvfg frfx mcig"
EMAIL_RECEIVER = "f2023-551@bnu.edu.pk"

# Track history for motion analysis
track_history = {}

# Virtual line configuration
line_x = 200  # x-coordinate for the virtual line
crossing_counts = {
    'left_to_right': 0,
    'right_to_left': 0
}
prev_centers = {}  # Store previous centers for line crossing detection

def send_email_alert(subject, body, image_path=None):
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time < ALERT_COOLDOWN:
        print(f"Alert cooldown active. Next alert in {ALERT_COOLDOWN - (current_time - last_alert_time):.1f} seconds.")
        return

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach image if provided
    if image_path and os.path.exists(image_path): # Check if file actually exists
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(image_path)) # Use basename for cleaner name
                msg.attach(image)
            print(f"Successfully attached image: {image_path}")
        except Exception as e:
            print(f"Failed to attach image {image_path}: {e}")
            image_path = None # Set to None so we don't try to remove a non-attached file
    elif image_path:
        print(f"Warning: Image file not found at {image_path}. Email will be sent without attachment.")


    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Email alert sent successfully.")
        last_alert_time = current_time
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        # Clean up the image file after sending (or attempting to send)
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Removed temporary image file: {image_path}")
            except Exception as e:
                print(f"Failed to remove temporary image file {image_path}: {e}")

# Define skeleton connections for drawing
skeleton = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# For motion analysis
STATIONARY_THRESHOLD = 5   # pixels movement threshold
STATIONARY_TIME = 5       # seconds - Set to 5 for quicker testing, then adjust to 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break

    current_time = time.time()
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- CRITICAL CHANGE: Make a copy of the raw frame BEFORE any processing ---
    # This ensures that 'original_frame' is pristine for cropping.
    original_frame_for_cropping = frame.copy()

    # Draw virtual counting line
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)

    # Object detection (used for DeepSORT tracking, as it provides good bounding boxes)
    results_detection = model(frame, verbose=False)[0] # Use 'model' for detection
    detections = []
    for box in results_detection.boxes:
        if int(box.cls[0]) == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
    
    # Segmentation (applied for visual overlay on the displayed frame, not directly for email crop)
    seg_results = seg_model(frame, verbose=False)[0]
    masks = {} # Initialize masks dictionary for the current frame
    if seg_results.masks is not None:
        for i, box in enumerate(seg_results.boxes):
            if int(box.cls[0]) == 0:  # class 0 = person
                # Ensure mask data exists for this index
                if i < len(seg_results.masks.data):
                    mask = seg_results.masks.data[i].cpu().numpy()
                    masks[i] = mask

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    face_detected_in_frame = False
    thumbs_up_detected = False

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)
        bbox = track.to_tlbr() # This gives the bounding box for the tracked object
        
        # Apply segmentation mask to the frame (for display purposes only)
        # This part applies the green overlay to the *displayed* frame
        if masks: # Only iterate if there are masks detected
            for i, mask in masks.items():
                # For robust mask application to a specific tracked person,
                # you would need to associate the mask's bounding box with the track's bounding box
                # (e.g., using IoU or proximity).
                # For simplicity here, we're just applying all person masks found by seg_model.
                # If you only want the mask on *tracked* people, you'd need more complex logic.
                
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                person_mask = (mask_resized > 0.5).astype(np.uint8) * 255

                colored_mask = np.zeros_like(frame)
                colored_mask[:, :, 1] = person_mask   # Green mask

                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)


        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bounding box coordinates are within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)

        # Calculate center point for line crossing detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Get previous center x for this track
        prev_x = prev_centers.get(track_id, center_x)

        # Line crossing detection
        if prev_x < line_x <= center_x:  # Left to right crossing
            crossing_counts['left_to_right'] += 1
            print(f"Object {track_id} crossed right. Total count: {crossing_counts['left_to_right']}")

            cropped = original_frame_for_cropping[y1:y2, x1:x2]
            if cropped.size > 0:
                img_path = f"crossed_person_{track_id}.jpg"
                cv2.imwrite(img_path, cropped)

                try:
                    emotion_result = DeepFace.analyze(cropped, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = emotion_result[0]['dominant_emotion']
                except Exception as e:
                    print(f"Emotion detection failed: {e}")
                    dominant_emotion = "unknown"

                subject = f"Person crossed line - ID {track_id}"
                body = f"Person ID {track_id} has crossed the line going right.\n"
                body += f"Total count: {crossing_counts['left_to_right']}\nDetected Emotion: {dominant_emotion.capitalize()}"

                send_email_alert(subject, body, image_path=img_path)

        elif prev_x > line_x >= center_x:  # Right to left crossing
            crossing_counts['right_to_left'] += 1
            print(f"Object {track_id} crossed left. Total count: {crossing_counts['right_to_left']}")

            cropped = original_frame_for_cropping[y1:y2, x1:x2]
            if cropped.size > 0:
                img_path = f"crossed_person_{track_id}.jpg"
                cv2.imwrite(img_path, cropped)

                try:
                    emotion_result = DeepFace.analyze(cropped, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = emotion_result[0]['dominant_emotion']
                except Exception as e:
                    print(f"Emotion detection failed: {e}")
                    dominant_emotion = "unknown"

                subject = f"Person crossed line - ID {track_id}"
                body = f"Person ID {track_id} has crossed the line going left.\n"
                body += f"Total count: {crossing_counts['right_to_left']}\nDetected Emotion: {dominant_emotion.capitalize()}"

                send_email_alert(subject, body, image_path=img_path)

        # Update previous center
        prev_centers[track_id] = center_x

        # Update track history for motion analysis
        center = (center_x, center_y)
        if track_id not in track_history:
            track_history[track_id] = {
                'positions': [center],
                'first_seen': current_time,
                'last_movement': current_time,
                'alert_sent': False,
                'last_bbox': (x1, y1, x2, y2) # Store last known bbox
            }
        else:
            # Calculate movement from last position
            last_pos = track_history[track_id]['positions'][-1]
            movement = np.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
            
            if movement > STATIONARY_THRESHOLD:
                track_history[track_id]['last_movement'] = current_time
                track_history[track_id]['alert_sent'] = False # Reset alert if moved
            
            track_history[track_id]['positions'].append(center)
            track_history[track_id]['last_bbox'] = (x1, y1, x2, y2) # Update bbox
            if len(track_history[track_id]['positions']) > 30: # Keep history short
                track_history[track_id]['positions'].pop(0)

        # Check for stationary condition
        time_tracked = current_time - track_history[track_id]['first_seen']
        stationary_duration = current_time - track_history[track_id]['last_movement']
        if (time_tracked > STATIONARY_TIME and
            stationary_duration > STATIONARY_TIME and
            not track_history[track_id]['alert_sent']):
            
            print(f"Person {track_id} stationary for {stationary_duration:.1f} seconds. Preparing alert.")
            
            # --- CRITICAL: Use the stored original frame and the track's bbox for cropping ---
            alert_x1, alert_y1, alert_x2, alert_y2 = track_history[track_id]['last_bbox']

            # Ensure valid cropping coordinates
            alert_x1, alert_y1 = max(0, alert_x1), max(0, alert_y1)
            alert_x2, alert_y2 = min(width-1, alert_x2), min(height-1, alert_y2)
            
            # Check if the bounding box has valid dimensions before cropping
            if alert_x2 > alert_x1 and alert_y2 > alert_y1:
                cropped_person_image = original_frame_for_cropping[alert_y1:alert_y2, alert_x1:alert_x2]
                
                # Save the cropped image
                if cropped_person_image.size > 0: # Ensure the cropped image is not empty
                    image_path = f"suspicious_person_{track_id}.jpg"
                    cv2.imwrite(image_path, cropped_person_image)
                    print(f"Saved cropped image: {image_path}")

                    try:
                        # Use DeepFace to analyze emotion
                        emotion_result = DeepFace.analyze(cropped_person_image, actions=['emotion'], enforce_detection=False)
                        dominant_emotion = emotion_result[0]['dominant_emotion']
                        email_subject = f"Suspicious Person Detected - ID {track_id}"
                        email_body = f"A person has been detected staying stationary for over {STATIONARY_TIME} seconds.\n\n"
                        email_body += f"Detected Emotion: {dominant_emotion.capitalize()}"
                        send_email_alert(email_subject, email_body, image_path=image_path)

                    except Exception as e:
                        print(f"Emotion detection failed: {e}")
                        dominant_emotion = "unknown"
                    # Send email with the cropped image
                    send_email_alert(
                        "Suspicious Activity Detected",
                        f"Person {track_id} has been stationary for over {STATIONARY_TIME} seconds. See the attached image.",
                        image_path=image_path
                    )
                    track_history[track_id]['alert_sent'] = True
                else:
                    print(f"Warning: Cropped image for person {track_id} is empty (size 0). Not saving/sending.")
            else:
                print(f"Warning: Invalid bounding box for cropping (x1:{alert_x1},y1:{alert_y1},x2:{alert_x2},y2:{alert_y2}). Not cropping/sending.")


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {track_id}", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, f"Person {track_id}", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Emotion detection + Face presence
        if (x2 - x1) > 50 and (y2 - y1) > 50:
            face_roi = frame[y1:y2, x1:x2]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                emotion = result.get('dominant_emotion', None)
                if emotion:
                    face_detected_in_frame = True
                    cv2.putText(frame, f"Emotion: {emotion}", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, f"Emotion: {emotion}", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # print(f"Person {track_id} Emotion: {emotion}") # Un-comment for more verbose output
            except Exception as e:
                # print(f"No face detected for person {track_id} in ROI: {e}") # Un-comment for more verbose output
                pass # Suppress frequent "No face detected" messages if not critical

    # Alert if person detected but no face detected
    # This alert also needs to capture an image of the person without a face.
    # We'll need to decide which tracked person to capture if multiple are present.
    # For simplicity, if *any* person is tracked and *no* face is detected overall,
    # it sends a generic alert. If you want to send an image of a *specific* person
    # without a face, you'd need to loop through tracks and find such a person.
    if len(tracks) > 0 and not face_detected_in_frame and (current_time - last_alert_time) >= ALERT_COOLDOWN:
        # Find a track to capture for the "no face" alert.
        # This is a simplification; you might want to pick the largest bbox or the oldest.
        person_for_no_face_alert = None
        for track in tracks:
            if track.is_confirmed():
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                if (x2 - x1) > 50 and (y2 - y1) > 50: # Ensure a reasonably sized bounding box
                    person_for_no_face_alert = (x1, y1, x2, y2, track.track_id)
                    break # Take the first suitable person

        if person_for_no_face_alert:
            x1, y1, x2, y2, track_id_for_alert = person_for_no_face_alert
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width-1, x2), min(height-1, y2)

            if x2 > x1 and y2 > y1:
                cropped_alert_image = original_frame_for_cropping[y1:y2, x1:x2]
                if cropped_alert_image.size > 0:
                    alert_image_path = f"no_face_person_{track_id_for_alert}.jpg"
                    cv2.imwrite(alert_image_path, cropped_alert_image)
                    subject = "Suspicious Activity Detected"
                    body = f"A person has been stationary.\nDetected Emotion: {dominant_emotion}"
                    send_email_alert(subject, body, image_path=image_path)
                else:
                    print("Warning: Cropped image for 'no face' alert was empty.")
            else:
                print("Warning: Invalid bounding box for 'no face' alert cropping.")
        else:
            send_email_alert("Suspicious Activity Detected", "Person detected but face is not visible or obscured.")


    # Pose estimation and drawing skeleton
    pose_results = pose_model(frame)[0]
    if pose_results.keypoints is not None and len(pose_results.keypoints.xy) > 0:
        for kp in pose_results.keypoints:
            # Check if keypoints are present for this 'kp'
            if kp.xy is not None and len(kp.xy) > 0:
                points = kp.xy[0]
                visibility = kp.conf[0] if kp.conf is not None else np.ones(len(points))

                for i, point in enumerate(points):
                    # Ensure the index 'i' is valid for visibility array
                    if i < len(visibility) and visibility[i] > 0.5:
                        x = int(point[0].item())
                        y = int(point[1].item())
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                for pt1_idx, pt2_idx in skeleton:
                    # Ensure indices are valid for 'points' and 'visibility'
                    if (pt1_idx < len(points) and pt2_idx < len(points) and
                        pt1_idx < len(visibility) and pt2_idx < len(visibility) and
                        visibility[pt1_idx] > 0.5 and visibility[pt2_idx] > 0.5):
                        pt1 = points[pt1_idx]
                        pt2 = points[pt2_idx]
                        x1 = int(pt1[0].item())
                        y1 = int(pt1[1].item())
                        x2 = int(pt2[0].item())
                        y2 = int(pt2[1].item())
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # print("Warning: No keypoints found for a pose detection result.") # Optional debug
                pass
    else:
        # print("No human pose detected in the frame.") # Optional debug
        pass # No pose detected, so skip drawing skeleton

    # MediaPipe Hands - Thumbs up detection
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            if thumb_tip.y < index_mcp.y: # A simple heuristic for thumbs up
                thumbs_up_detected = True
                cv2.putText(frame, "Thumbs Up Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if thumbs_up_detected:
        print("Thumbs up gesture detected.")

    # Clean up old tracks from track_history that are no longer confirmed by DeepSORT
    # Or, if they haven't been seen for a while (e.g., 5-10 seconds after DeepSORT max_age)
    # This part needs to be more robust. For simplicity, we'll keep tracking as long as DeepSORT confirms.

    # Display crossing counts
    cv2.putText(frame, f"L->R: {crossing_counts['left_to_right']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"R->L: {crossing_counts['right_to_left']}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display
    cv2.imshow("AI Surveillance System", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()