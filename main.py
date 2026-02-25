import subprocess
import time
import sys

print("🚀 [System] Start Teleoperation System")

# 1. Isaac Sim Reciever
print("⏳ [System] 1. Launching Isaac Sim (Receiver) Environment")
receiver_process = subprocess.Popen([sys.executable, "receiver_isaac.py"])

time.sleep(2)

# 2. RealSense
print("📷 [System] 2. Launching RealSense Camera & MediaPipe (Sender)")
sender_process = subprocess.Popen([sys.executable, "sender.py"])

try:
    # Wait until the two processes end
    receiver_process.wait()
    sender_process.wait()

except KeyboardInterrupt:
    # Shut down by user
    print("\n🛑 [System] ShutDown by User")
    
    # Send termination command to the two sub-processes
    sender_process.terminate()
    receiver_process.terminate()
    
    # Wait until they completely turn off
    sender_process.wait()
    receiver_process.wait()
    
    print("✅ [System] All processes have been safely terminated.")