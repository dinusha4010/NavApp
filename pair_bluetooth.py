#!/usr/bin/env python3
import os
import subprocess
import time
import sys

def run_command(cmd, capture=True):
    """Run a shell command and return the output"""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, check=False, 
                                   capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=False)
            return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None

def start_bluetooth_service():
    """Make sure Bluetooth service is running"""
    print("Starting Bluetooth service...")
    run_command("sudo systemctl enable bluetooth", capture=False)
    run_command("sudo systemctl start bluetooth", capture=False)
    time.sleep(2)
    
    # Check if service is running
    status = run_command("systemctl is-active bluetooth")
    if status == "active":
        print("Bluetooth service is active")
        return True
    else:
        print("Failed to start Bluetooth service")
        return False

def scan_for_speakers():
    """Scan for available Bluetooth devices"""
    print("\nScanning for Bluetooth devices...")
    print("Wait for at least 10 seconds for devices to appear")
    
    # Start scanning
    run_command("bluetoothctl scan on", capture=False)
    
    # Wait for some time to discover devices
    for i in range(10):
        sys.stdout.write(f"\rScanning... {i+1}/10 seconds")
        sys.stdout.flush()
        time.sleep(1)
    print("\n")
    
    # Get list of available devices
    devices_raw = run_command("bluetoothctl devices")
    
    if not devices_raw:
        print("No devices found. Make sure your speaker is in pairing mode.")
        return []
    
    devices = []
    for line in devices_raw.splitlines():
        if line.strip():
            parts = line.split(" ", 2)
            if len(parts) >= 3:
                mac = parts[1]
                name = parts[2]
                devices.append((mac, name))
    
    return devices

def pair_with_speaker(mac_address):
    """Pair with the selected Bluetooth speaker"""
    print(f"\nPairing with {mac_address}...")
    run_command(f"bluetoothctl pair {mac_address}", capture=False)
    time.sleep(2)
    
    # Trust the device so it automatically connects in future
    print(f"Trusting {mac_address}...")
    run_command(f"bluetoothctl trust {mac_address}", capture=False)
    time.sleep(1)
    
    # Connect to the device
    print(f"Connecting to {mac_address}...")
    run_command(f"bluetoothctl connect {mac_address}", capture=False)
    time.sleep(2)
    
    # Check if it's connected
    info = run_command(f"bluetoothctl info {mac_address}")
    if "Connected: yes" in info:
        print("Successfully connected!")
        return True
    else:
        print("Failed to connect. Try again or check speaker pairing mode.")
        return False

def setup_audio():
    """Make sure PulseAudio is running and set default sink"""
    # Start PulseAudio if not running
    run_command("pulseaudio --start", capture=False)
    time.sleep(2)
    
    # List available sinks
    print("\nAvailable audio output devices:")
    sinks = run_command("pactl list short sinks")
    if sinks:
        print(sinks)
    else:
        print("No audio sinks found")
    
    # List Bluetooth devices connected for audio
    print("\nConnected Bluetooth audio devices:")
    cards = run_command("pactl list cards | grep -A 20 bluez")
    if cards:
        print(cards)
    else:
        print("No Bluetooth audio devices found connected to PulseAudio")

def test_speaker():
    """Test the speaker with espeak"""
    print("\nTesting speaker with espeak...")
    run_command('espeak "Testing Bluetooth speaker connection" -ven+f3 -k5 -s150', capture=False)
    
    response = input("\nDid you hear the test message? (y/n): ")
    if response.lower() == 'y':
        print("Great! Speaker is working correctly.")
        return True
    else:
        print("Speaker test failed. Make sure the device is connected and volume is up.")
        return False

def main():
    print("\n=== Bluetooth Speaker Setup Tool ===\n")
    
    # 1. Make sure Bluetooth service is running
    if not start_bluetooth_service():
        print("Cannot proceed without Bluetooth service")
        return
    
    # 2. Scan for available devices
    devices = scan_for_speakers()
    
    if not devices:
        print("No devices found. Please try again later.")
        return
    
    # 3. Display available devices
    print("\nAvailable Bluetooth devices:")
    for i, (mac, name) in enumerate(devices):
        print(f"{i+1}. {name} ({mac})")
    
    # 4. Let user select a device
    try:
        choice = int(input("\nSelect a device to pair (enter number): ")) - 1
        if choice < 0 or choice >= len(devices):
            print("Invalid selection")
            return
    except ValueError:
        print("Please enter a valid number")
        return
    
    mac_address = devices[choice][0]
    
    # 5. Pair with the device
    if pair_with_speaker(mac_address):
        print("\nPairing successful!")
    else:
        print("\nPairing process failed.")
        return
    
    # 6. Setup audio
    setup_audio()
    
    # 7. Test the speaker
    test_speaker()
    
    print("\nSetup complete! Instructions for using with the camera app:")
    print("1. The app will now automatically use the Bluetooth speaker")
    print("2. To manually test: python app.py")
    print("3. To manually disconnect: bluetoothctl disconnect", mac_address)

if __name__ == "__main__":
    main() 