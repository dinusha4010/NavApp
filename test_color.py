from picamera2 import Picamera2
import cv2
import time

print("Starting camera test with RGB format and auto white balance...")

picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)},
    controls={"AwbEnable": True, "AwbMode": 0}  # Auto white balance
)
picam2.configure(preview_config)

print("Starting camera...")
picam2.start()
print("Camera started successfully")

# Take 5 test frames
print("Taking test frames...")
for i in range(5):
    frame = picam2.capture_array()
    print(f"Frame {i+1} shape: {frame.shape}")
    
    # Save one frame as test
    if i == 0:
        cv2.imwrite("/tmp/test_rgb.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
        print("Saved test frame to /tmp/test_rgb.jpg")
    
    time.sleep(0.5)

print("Stopping camera...")
picam2.stop()
print("Camera test complete.") 