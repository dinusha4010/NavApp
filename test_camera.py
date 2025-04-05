from picamera2 import Picamera2
import cv2
import time

print("Starting camera test...")

picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": 'BGR888', "size": (640, 480)})
picam2.configure(preview_config)

print("Starting camera...")
picam2.start()
print("Camera started successfully")

# Take 10 test frames
print("Taking test frames...")
for i in range(10):
    frame = picam2.capture_array()
    print(f"Frame {i+1} shape: {frame.shape}")
    
    # Save one frame as test
    if i == 0:
        cv2.imwrite("/tmp/test_frame.jpg", frame)
        print("Saved test frame to /tmp/test_frame.jpg")
    
    time.sleep(0.5)

print("Stopping camera...")
picam2.stop()
print("Camera test complete.") 