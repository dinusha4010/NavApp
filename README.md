# GPS Location Tracker with Camera

A Flask web application that shows your current GPS location on a Mapbox map along with a real-time camera feed from your Raspberry Pi camera.

## Prerequisites

- Raspberry Pi 5
- GPS device connected to `/dev/ttyACM0`
- Pi Camera v2 connected and enabled
- `gpsd` installed

## Setup

1. Install required packages:
```
sudo apt-get update
sudo apt-get install -y python3-pip python3-full gpsd gpsd-clients libopencv-dev python3-opencv
```

2. Create a virtual environment (required on Raspberry Pi OS):
```
mkdir -p ~/python_envs
python3 -m venv ~/python_envs/gps_tracker
source ~/python_envs/gps_tracker/bin/activate
```

3. Install Python dependencies in the virtual environment:
```
pip install -r requirements.txt
```

4. Make sure your camera is enabled:
```
# Check if camera is enabled
vcgencmd get_camera

# If needed, enable the camera
sudo raspi-config
# Navigate to Interface Options > Camera > Enable
```

5. Start the GPS daemon:
```
sudo gpsd /dev/ttyACM0 -F /var/run/gpsd.sock
```

6. Run the Flask application (with activated virtual environment):
```
python app.py
```

7. Open the web application in your browser:
```
http://[your-raspberry-pi-ip]:5000
```

## Testing

You can verify your GPS is working properly with:
```
cgps -s
```

You can test your camera independently with:
```
libcamera-still -o test_image.jpg
```

## Features

- Real-time GPS position tracking
- Interactive Mapbox map
- Live camera feed from Raspberry Pi camera
- Position updates every 3 seconds
- Responsive design for mobile and desktop viewing

## Troubleshooting

### Camera Issues

If the camera feed doesn't appear:
- Make sure the camera is properly connected
- Check that the camera is enabled in raspi-config
- Verify camera permissions: `ls -la /dev/video0`
- Try testing with a direct camera command: `libcamera-still -o test.jpg`

### Package Installation

If you prefer not to use a virtual environment, you can add the `--break-system-packages` flag to pip:
```
pip3 install -r requirements.txt --break-system-packages
```
However, this is not recommended as it may interfere with system packages. 