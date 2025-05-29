import cv2
from deepface import DeepFace
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0)

# Define colors
box_color = (0, 140, 255)      # Bright orange
text_color = (255, 255, 255)   # White
text_bg_color = (0, 0, 0)      # Black outline
tracking_colors = [
    (0, 255, 0),   # Green
    (255, 0, 0),   # Blue
    (0, 0, 255),   # Red
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255), # Yellow
    (255, 165, 0), # Orange
    (128, 0, 128)  # Purple
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Run YOLOv8 detection
    results = model(frame, verbose=False)[0]
    
    # Prepare detections for DeepSORT
    detections = []
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # Only process 'person' class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
    
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
            
        # Get track ID and convert to integer
        try:
            track_id = str(track.track_id)
            color_idx = hash(track_id) % len(tracking_colors)
            color = tracking_colors[color_idx]
        except:
            track_id = "0"
            color = (0, 255, 0)  # Fallback color
            
        # Get bounding box
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID
        label = f"Person {track_id}"
        cv2.putText(frame, label, (x1, max(y1-10, 20)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bg_color, 3)
        cv2.putText(frame, label, (x1, max(y1-10, 20)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Emotion detection for sufficiently large boxes
        if (x2 - x1) > 50 and (y2 - y1) > 50:
            face_roi = frame[y1:y2, x1:x2]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], 
                                        enforce_detection=False, silent=True)
                emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
                
                # Display emotion
                text_y = min(y2 + 25, height - 10)
                cv2.putText(frame, f"Emotion: {emotion}", (x1, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_bg_color, 3)
                cv2.putText(frame, f"Emotion: {emotion}", (x1, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            except Exception as e:
                print(f"Emotion detection failed for track {track_id}: {str(e)}")

    # Display frame
    cv2.imshow('AI Surveillance System', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()