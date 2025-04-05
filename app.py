from flask import Flask, render_template, jsonify, Response
import gpsd
import time
import cv2
import numpy as np
import threading
import os
import subprocess
import io
from pathlib import Path
import sys

# Print Python version and environment info
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")

# Explicitly disable debug for production use
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Global flag for camera availability
camera_type = "none"

# YOLO model configuration
YOLO_MODEL_READY = False
DETECT_HUMANS = True
DETECT_ROAD_SIGNS = True  # Enable road sign detection
DETECT_CROSSINGS = True   # Enable pedestrian crossing detection
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Audio alert configuration
AUDIO_ENABLED = True
AUDIO_COOLDOWN = 3.0  # Shorter cooldown for more responsive alerts
last_audio_alert = 0
last_detection_type = None  # Track last detection to avoid repetition

# Tracking for crossing detection
pedestrian_tracks = {}  # Store pedestrian tracks to detect crossing
track_id_counter = 0

# Road sign classes to detect
ROAD_SIGN_CLASSES = [
    "stop sign", "traffic light", "car", "truck", "motorcycle", "bicycle"
]

# Try to import picamera2
try:
    from picamera2 import Picamera2
    camera_type = "picamera2"
    print("Using picamera2 for camera access")
except ImportError:
    print("picamera2 not available, trying alternative methods")
    # Check if libcamera-still is available
    try:
        result = subprocess.run(["which", "libcamera-still"], 
                               capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            camera_type = "libcamera"
            print("Using libcamera-still for camera access")
        else:
            print("WARNING: libcamera-still not found")
    except Exception as e:
        print(f"Error checking for libcamera: {e}")

# Check if espeak is installed
try:
    result = subprocess.run(["which", "espeak"], 
                           capture_output=True, text=True, timeout=2)
    if result.returncode == 0:
        print("espeak found - audio alerts enabled")
    else:
        print("WARNING: espeak not found - install with 'sudo apt-get install espeak'")
        AUDIO_ENABLED = False
except Exception as e:
    print(f"Error checking for espeak: {e}")
    AUDIO_ENABLED = False

app = Flask(__name__)
app.config['DEBUG'] = False

# Mapbox API key
MAPBOX_API_KEY = "pk.eyJ1IjoiZGludXNoYTQwMTAiLCJhIjoiY204ejFxYWZoMDVqcDJpcjBweDNscTEwaSJ9.PY8-geSMKAx7PDikoDubtA"

# Global variables for camera
camera_active = False
output_frame = None
lock = threading.Lock()
picam2 = None

# YOLO model globals
net = None
output_layers = None
classes = None

def initialize_yolo():
    """Initialize YOLO model for human detection"""
    global net, output_layers, classes, YOLO_MODEL_READY
    
    print("Initializing YOLO model for human and road sign detection...")
    
    try:
        # Load COCO class names - in COCO dataset, person class is at index 0
        classes_file = "models/coco.names"
        if not os.path.exists(classes_file):
            # Create the classes file with COCO classes
            with open(classes_file, "w") as f:
                f.write("person\ncar\nbicycle\nmotorcycle\nairplane\nbus\ntrain\ntruck\nboat\n")
                f.write("traffic light\nfire hydrant\nstop sign\nparking meter\nbench\nbird\ncat\ndog\n")
                f.write("horse\nsheep\ncow\nelephant\nbear\nzebra\ngiraffe\nbackpack\numbrella\n")
                f.write("handbag\ntie\nsuitcase\nfrisbee\nskis\nsnowboard\nsports ball\nkite\n")
                f.write("baseball bat\nbaseball glove\nskateboard\nsurfboard\ntennis racket\nbottle\n")
                f.write("wine glass\ncup\nfork\nknife\nspoon\nbowl\nbanana\napple\nsandwich\norange\n")
                f.write("broccoli\ncarrot\nhot dog\npizza\ndonut\ncake\nchair\ncouch\npotted plant\nbed\n")
                f.write("dining table\ntoilet\ntv\nlaptop\nmouse\nremote\nkeyboard\ncell phone\nmicrowave\n")
                f.write("oven\ntoaster\nsink\nrefrigerator\nbook\nclock\nvase\nscissors\nteddy bear\nhair drier\ntoothbrush")
            
        # Load COCO class names
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(classes)} classes, person is at index {classes.index('person')}")
        
        # Try to use a small YOLOv4 model if available (less resource intensive)
        yolo_cfg = "models/yolov4-tiny.cfg"
        yolo_weights = "models/yolov4-tiny.weights"
        
        # Create model directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Download model files if they don't exist
        if not os.path.exists(yolo_cfg) or not os.path.exists(yolo_weights):
            print("Downloading YOLO model files...")
            subprocess.run([
                "wget", "https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4-tiny.cfg",
                "-O", yolo_cfg
            ], check=True)
            subprocess.run([
                "wget", "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
                "-O", yolo_weights
            ], check=True)
            print("Model files downloaded successfully")
        
        # Load the YOLO network
        net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
        
        # Run on CPU for better compatibility
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        try:
            # OpenCV 4.5.4+
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            # Earlier OpenCV versions
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            
        print(f"YOLO model initialized with {len(output_layers)} output layers")
        
        # Print which road sign classes we'll detect
        road_sign_indices = []
        for road_sign in ROAD_SIGN_CLASSES:
            if road_sign in classes:
                road_sign_indices.append(classes.index(road_sign))
        print(f"Will detect road signs: {ROAD_SIGN_CLASSES}")
        print(f"Road sign classes at indices: {road_sign_indices}")
        
        YOLO_MODEL_READY = True
        return True
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        return False

def is_crossing_road(box, frame_width, frame_height):
    """Determine if a pedestrian is likely crossing a road based on position"""
    x, y, w, h = box
    
    # Crossing is more likely if pedestrian is in the middle portion of the frame
    # and in the lower half (closer to the camera in a typical road setup)
    
    center_x = x + w/2
    center_y = y + h/2
    
    # Check if in the middle horizontal third of the frame 
    # and in the bottom half vertically
    in_middle_x = frame_width/3 < center_x < (2*frame_width/3)
    in_bottom_half = center_y > frame_height/2
    
    return in_middle_x and in_bottom_half

def track_pedestrian(pedestrian_id, box, frame_shape):
    """Track pedestrian movement to detect crossing patterns"""
    global pedestrian_tracks
    
    # Extract current position
    x, y, w, h = box
    center_x = x + w/2
    frame_width = frame_shape[1]
    
    # If this pedestrian is new, initialize tracking
    if pedestrian_id not in pedestrian_tracks:
        pedestrian_tracks[pedestrian_id] = {
            "positions": [(center_x, y + h)],  # Track bottom center point
            "crossing_detected": False,
            "last_seen": time.time()
        }
        return False
    
    # Add new position
    pedestrian_tracks[pedestrian_id]["positions"].append((center_x, y + h))
    pedestrian_tracks[pedestrian_id]["last_seen"] = time.time()
    
    # Keep only the last 10 positions
    if len(pedestrian_tracks[pedestrian_id]["positions"]) > 10:
        pedestrian_tracks[pedestrian_id]["positions"] = pedestrian_tracks[pedestrian_id]["positions"][-10:]
    
    # If we've already detected crossing for this pedestrian, don't repeat
    if pedestrian_tracks[pedestrian_id]["crossing_detected"]:
        return False
    
    # Need at least 5 positions to analyze movement
    if len(pedestrian_tracks[pedestrian_id]["positions"]) < 5:
        return False
    
    # Check if the pedestrian is in the middle third of the frame
    in_middle = frame_width/3 < center_x < (2*frame_width/3)
    
    # Calculate horizontal movement
    positions = pedestrian_tracks[pedestrian_id]["positions"]
    x_positions = [pos[0] for pos in positions]
    
    # Check if there's significant horizontal movement
    x_min, x_max = min(x_positions), max(x_positions)
    significant_movement = (x_max - x_min) > frame_width * 0.1  # At least 10% of frame width
    
    # Mark as crossing if in middle and showing significant movement
    if in_middle and significant_movement:
        pedestrian_tracks[pedestrian_id]["crossing_detected"] = True
        return True
    
    return False

def detect_objects(frame):
    """Detect humans, road signs, and crossing pedestrians in the frame"""
    global net, output_layers, classes, YOLO_MODEL_READY, track_id_counter, pedestrian_tracks
    
    if not YOLO_MODEL_READY:
        return frame, []
    
    # Clean up old pedestrian tracks
    current_time = time.time()
    for pid in list(pedestrian_tracks.keys()):
        if current_time - pedestrian_tracks[pid]["last_seen"] > 5.0:  # Remove after 5 seconds
            del pedestrian_tracks[pid]
    
    detections = []  # Store all detections to announce
    
    try:
        # Ensure we have 3 channels (BGR) for OpenCV DNN
        if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] != 3:
            # Create a 3-channel BGR image from the data
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame_rgb
            
        height, width, channels = frame.shape
        
        if channels != 3:
            # Force creation of a 3-channel image as fallback
            bgr_frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Copy available channels
            for i in range(min(channels, 3)):
                bgr_frame[:,:,i] = frame[:,:,i]
            frame = bgr_frame
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set the blob as input to the network
        net.setInput(blob)
        
        # Run forward pass
        outs = net.forward(output_layers)
        
        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each detection
        person_index = classes.index('person')
        detected = False
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for relevant classes with suitable confidence threshold
                is_person = class_id == person_index and DETECT_HUMANS
                is_road_sign = (
                    DETECT_ROAD_SIGNS and
                    classes[class_id] in ROAD_SIGN_CLASSES
                )
                
                # Detect if confidence threshold is met for any valid class
                if (is_person or is_road_sign) and confidence > 0.3:
                    detected = True
                    # YOLO returns center, width, height in relative coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        if not detected:
            # If nothing detected, just add info on the frame
            cv2.putText(frame, "No objects detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, []
        
        # Apply non-maximum suppression to suppress overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        
        # Process and draw detections
        font = cv2.FONT_HERSHEY_SIMPLEX
        crossing_detected = False
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]
                
                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                
                # Get class name
                class_name = classes[class_id]
                
                # Different processing based on class
                if class_name == "person":
                    # Check for road crossing
                    is_crossing = False
                    
                    if DETECT_CROSSINGS:
                        # Assign tracking ID to this pedestrian
                        track_id_counter += 1
                        pedestrian_id = f"ped_{track_id_counter}"
                        
                        # Check if pedestrian appears to be crossing
                        is_crossing = is_crossing_road([x, y, w, h], width, height)
                        
                        # Track pedestrian for crossing detection
                        is_tracking_crossing = track_pedestrian(pedestrian_id, [x, y, w, h], frame.shape)
                        
                        is_crossing = is_crossing or is_tracking_crossing
                    
                    if is_crossing:
                        # Pedestrian is crossing - high alert!
                        label = f"CROSSING! {confidence:.2f}"
                        color = (0, 0, 255)  # Red for crossing
                        detections.append(("pedestrian_crossing", confidence))
                        crossing_detected = True
                    else:
                        # Normal pedestrian
                        label = f"Person {confidence:.2f}"
                        color = (0, 255, 0)  # Green for normal pedestrian
                        detections.append(("person", confidence))
                
                elif class_name in ROAD_SIGN_CLASSES:
                    # Road sign or vehicle detection
                    label = f"{class_name.upper()} {confidence:.2f}"
                    
                    # Different colors for different sign types
                    if class_name == "stop sign":
                        color = (0, 0, 255)  # Red for stop signs
                    elif class_name == "traffic light":
                        color = (0, 255, 255)  # Yellow for traffic lights
                    elif class_name in ["car", "truck", "bus"]:
                        color = (255, 0, 0)  # Blue for vehicles
                    else:
                        color = (255, 0, 255)  # Magenta for other signs
                    
                    detections.append((class_name, confidence))
                
                else:
                    # Other classes (shouldn't happen with our filtering)
                    label = f"{class_name} {confidence:.2f}"
                    color = (255, 255, 0)
                
                # Draw the box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), font, 0.5, color, 2)
                
                # Add extra warning for crossing
                if crossing_detected:
                    cv2.putText(frame, "WARNING: PEDESTRIAN CROSSING", 
                               (width//2 - 150, 30), font, 0.7, (0, 0, 255), 2)
        
        return frame, detections
    
    except Exception as e:
        print(f"Error in object detection: {e}")
        # Add error info to the frame
        cv2.putText(frame, f"Detection error: {str(e)}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame, []

def initialize_camera():
    global camera_active, picam2, camera_type
    
    try:
        # Kill any existing camera processes that might conflict
        try:
            subprocess.run(["sudo", "pkill", "-f", "libcamera"], 
                          capture_output=True, timeout=3)
            time.sleep(1)
        except Exception as e:
            print(f"Note: pkill command failed: {e}")
        
        # Initialize based on available camera type
        if camera_type == "picamera2":
            # Initialize with picamera2 API
            picam2 = Picamera2()
            
            # Configure with BGR format (3 channels) to match what YOLO expects
            preview_config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "BGR888"}
            )
            picam2.configure(preview_config)
            
            # Start the camera
            picam2.start()
            
            # Take a test capture
            frame = picam2.capture_array()
            if frame is not None and len(frame.shape) == 3:
                print(f"Camera test successful: {frame.shape}")
                # Verify number of channels
                if frame.shape[2] != 3:
                    print(f"Warning: Camera produced {frame.shape[2]} channels instead of 3")
                camera_active = True
                return True
            else:
                print("Camera initialization failed: invalid frame")
                return False
            
        elif camera_type == "libcamera":
            # Test the camera with libcamera-still
            try:
                print("Testing camera with libcamera-still...")
                subprocess.run([
                    "libcamera-still", 
                    "-n", "-o", "/tmp/test.jpg",
                    "--immediate", 
                    "--width", "320", "--height", "240",
                    "--timeout", "500"
                ], timeout=2, capture_output=True)
                
                if os.path.exists("/tmp/test.jpg"):
                    # Read the test image to verify
                    test_img = cv2.imread("/tmp/test.jpg")
                    if test_img is not None:
                        print("Camera test successful with libcamera-still")
                        camera_active = True
                        return True
                
                print("Camera test failed: could not capture image")
                return False
            except Exception as e:
                print(f"Camera test failed: {e}")
                return False
        else:
            print("No supported camera method available")
            return False
            
    except Exception as e:
        print(f"Error initializing camera: {e}")
        camera_active = False
        return False

def get_camera_frame():
    global camera_active, output_frame, lock, picam2, camera_type
    
    frame_count = 0
    temp_jpg = "/tmp/camera_frame.jpg"
    last_capture_time = 0
    capture_interval = 0.05  # 20 FPS target for picamera2
    
    while True:
        if not camera_active:
            time.sleep(0.5)
            continue
        
        try:
            current_time = time.time()
            
            # Rate limiting for all capture methods
            if current_time - last_capture_time < capture_interval:
                time.sleep(0.01)  # Short sleep
                continue
            
            # Capture based on available method
            if camera_type == "picamera2" and picam2 is not None:
                # Direct capture with picamera2
                try:
                    # Get frame directly with native colors
                    frame = picam2.capture_array()
                    last_capture_time = time.time()
                    
                    if frame is not None:
                        # Apply detection on every frame
                        detection_start = time.time()
                        frame, detections = detect_objects(frame)
                        detection_time = time.time() - detection_start
                        
                        # Process detections for audio alerts
                        if detections:
                            process_detections_for_audio(detections)
                            
                        # Add detection time info
                        cv2.putText(frame, f"Detection: {detection_time:.3f}s", 
                                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Add timestamp and frame info
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, timestamp, (5, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Add frame shape info to diagnose any format issues
                        shape_info = f"Frame: {frame.shape}"
                        cv2.putText(frame, shape_info, (5, frame.shape[0] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Update the global frame with threading lock
                        with lock:
                            output_frame = frame.copy()
                        
                        frame_count += 1
                        if frame_count % 50 == 0:
                            print(f"Processed frame {frame_count}")
                except Exception as e:
                    print(f"Error capturing with picamera2: {e}")
                    time.sleep(0.5)
                    
            elif camera_type == "libcamera":
                # Use libcamera-still for slower but reliable capture
                # Limit to max 2 FPS to avoid overwhelming the system
                if current_time - last_capture_time < 0.5:
                    time.sleep(0.1)
                    continue
                    
                try:
                    # Capture a new frame
                    subprocess.run([
                        "libcamera-still", 
                        "-n", "-o", temp_jpg,
                        "--immediate", 
                        "--width", "320", "--height", "240",
                        "--quality", "70",
                        "--timeout", "200"
                    ], timeout=1, capture_output=True)
                    
                    last_capture_time = time.time()
                    
                    if os.path.exists(temp_jpg):
                        frame = cv2.imread(temp_jpg)
                        if frame is not None:
                            # Apply detection
                            frame, detections = detect_objects(frame)
                            
                            # Process detections for audio alerts
                            if detections:
                                process_detections_for_audio(detections)
                            
                            # Add timestamp
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            cv2.putText(frame, timestamp, (5, frame.shape[0] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            with lock:
                                output_frame = frame.copy()
                            
                            frame_count += 1
                            if frame_count % 10 == 0:
                                print(f"Processed frame {frame_count}")
                except Exception as e:
                    print(f"Error in libcamera capture: {e}")
                    time.sleep(0.5)
            else:
                # No camera method available, just wait
                time.sleep(1)
        
        except Exception as e:
            print(f"Error in camera frame capture: {e}")
            time.sleep(1)

def process_detections_for_audio(detections):
    """Process detected objects and trigger appropriate audio alerts"""
    if not detections:
        return
    
    # Sort detections by priority and confidence
    # Higher priority items: crossing pedestrians, stop signs, traffic lights
    priority_detections = []
    normal_detections = []
    
    for det_type, confidence in detections:
        if det_type == "pedestrian_crossing":
            priority_detections.append((det_type, confidence, 0))  # Highest priority
        elif det_type == "stop sign":
            priority_detections.append((det_type, confidence, 1))
        elif det_type == "traffic light":
            priority_detections.append((det_type, confidence, 2))
        elif det_type in ["car", "truck", "bus", "motorcycle"]:
            priority_detections.append((det_type, confidence, 3))
        else:
            normal_detections.append((det_type, confidence, 4))
    
    # Sort by priority first, then confidence
    priority_detections.sort(key=lambda x: (x[2], -x[1]))
    normal_detections.sort(key=lambda x: -x[1])
    
    sorted_detections = priority_detections + normal_detections
    
    # Announce the most important detection
    if sorted_detections:
        top_detection = sorted_detections[0]
        announce_detection(top_detection[0])

def announce_detection(detection_type):
    """Announce the detected object with appropriate message"""
    global last_detection_type
    
    # Skip if it's the same as the last detection to avoid repetition
    if detection_type == last_detection_type:
        return
    
    # Map detection types to spoken messages
    messages = {
        "pedestrian_crossing": "Warning! Pedestrian crossing road",
        "stop sign": "Stop sign ahead",
        "traffic light": "Traffic light ahead",
        "person": "Human detected",
        "car": "Vehicle detected",
        "truck": "Large vehicle ahead",
        "bus": "Bus detected",
        "motorcycle": "Motorcycle detected",
        "bicycle": "Bicycle detected"
    }
    
    # Get appropriate message or use detection type directly
    message = messages.get(detection_type, f"{detection_type} detected")
    
    # Speak the message
    speak_text(message)
    
    # Update last detection type
    last_detection_type = detection_type

def speak_text(text):
    """Speak the provided text through espeak"""
    global last_audio_alert, AUDIO_ENABLED, AUDIO_COOLDOWN
    
    # Check if audio is enabled and cooldown period has passed
    current_time = time.time()
    if not AUDIO_ENABLED or current_time - last_audio_alert < AUDIO_COOLDOWN:
        return False
        
    try:
        # Update last alert time
        last_audio_alert = current_time
        
        # Run espeak with the text
        print(f"Speaking: {text}")
        subprocess.Popen(["espeak", "-ven+f3", "-k5", "-s150", text], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"Error speaking text: {e}")
        return False

def generate_frames():
    global output_frame, lock
    
    while True:
        # Get the current frame with a lock
        with lock:
            if output_frame is not None:
                frame = output_frame.copy()
            else:
                frame = None
        
        # If no frame is available, generate a placeholder
        if frame is None:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for camera...", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        try:
            # Encode frame as JPEG
            result, encoded_frame = cv2.imencode('.jpg', frame)
            if not result:
                continue
                
            # Yield the frame in MJPEG format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encoded_frame) + b'\r\n')
            
            # Rate limiting
            time.sleep(0.05)  # Max 20 FPS for the browser stream
            
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            time.sleep(0.5)

@app.route('/')
def index():
    return render_template('index.html', mapbox_api_key=MAPBOX_API_KEY)

@app.route('/get_location')
def get_location():
    try:
        # Connect to the local gpsd
        gpsd.connect()
        
        # Get the current position
        packet = gpsd.get_current()
        
        # Extract latitude and longitude
        lat = packet.lat
        lon = packet.lon
        
        print(f"GPS data retrieved: lat={lat}, lon={lon}")
        
        return jsonify({
            'success': True,
            'latitude': lat,
            'longitude': lon,
            'time': time.time()
        })
    except Exception as e:
        print(f"Error getting GPS location: {str(e)}")
        # Return a fallback for testing if needed
        return jsonify({
            'success': False,
            'error': str(e),
            'errorType': str(type(e).__name__)
        })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_status')
def camera_status():
    global camera_active, output_frame
    # Include frame check to provide more accurate status
    if camera_active and output_frame is not None:
        return jsonify({
            'status': 'active',
            'method': camera_type
        })
    else:
        return jsonify({
            'status': 'inactive',
            'method': camera_type
        })

def cleanup():
    global picam2, camera_active
    
    print("Cleaning up resources...")
    camera_active = False
    
    try:
        # Stop picamera if it's running
        if camera_type == "picamera2" and picam2 is not None:
            picam2.stop()
            print("Camera stopped")
            
        # Clean up temporary files
        for temp_file in ["/tmp/camera_frame.jpg", "/tmp/test.jpg"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    except Exception as e:
        print(f"Error during cleanup: {e}")

def setup_bluetooth():
    """Setup Bluetooth connection to speaker"""
    if not AUDIO_ENABLED:
        return False
        
    try:
        print("Setting up Bluetooth audio...")
        
        # Check if bluetooth service is running
        subprocess.run(["systemctl", "is-active", "bluetooth"], 
                      capture_output=True, check=True, timeout=5)
        
        print("Bluetooth service is running")
        return True
    except Exception as e:
        print(f"Error setting up Bluetooth: {e}")
        print("To setup Bluetooth speaker manually:")
        print("1. Run 'sudo bluetoothctl'")
        print("2. In bluetoothctl: 'scan on'")
        print("3. Find your speaker's MAC address")
        print("4. In bluetoothctl: 'pair XX:XX:XX:XX:XX:XX' (replace with your speaker's MAC)")
        print("5. In bluetoothctl: 'connect XX:XX:XX:XX:XX:XX'")
        print("6. In bluetoothctl: 'trust XX:XX:XX:XX:XX:XX'")
        return False

if __name__ == '__main__':
    # Setup Bluetooth for audio
    bluetooth_setup = setup_bluetooth()
    print(f"Bluetooth setup {'successful' if bluetooth_setup else 'failed'}")
    
    # Initialize YOLO model for human detection
    yolo_init_success = initialize_yolo()
    print(f"YOLO initialization {'successful' if yolo_init_success else 'failed'}")
    
    # Start camera thread
    print("Initializing camera...")
    camera_init_success = initialize_camera()
    print(f"Camera initialization {'successful' if camera_init_success else 'failed'}")
    
    if camera_init_success:
        # Start the camera frame capture thread
        t = threading.Thread(target=get_camera_frame)
        t.daemon = True
        t.start()
        print("Camera frame capture thread started")
    else:
        print("WARNING: Camera initialization failed, proceeding without camera")
    
    # Register cleanup function
    import atexit
    atexit.register(cleanup)
    
    # Start the Flask app
    print(f"Starting Flask app on http://0.0.0.0:5000 with camera type: {camera_type}")
    app.run(host='0.0.0.0', port=5000, threaded=True) 