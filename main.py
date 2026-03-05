import subprocess
import sys

print("Start Teleoperation System (Vision Pro)")

# Single process: receiver_isaac.py now talks directly to Vision Pro
print("Launching Isaac Sim + Vision Pro Teleoperation...")
receiver_process = subprocess.Popen([sys.executable, "receiver_isaac.py"])

try:
    receiver_process.wait()

except KeyboardInterrupt:
    print("\nShutDown by User")
    receiver_process.terminate()
    receiver_process.wait()
    
    print("✅ [System] All processes have been safely terminated.")