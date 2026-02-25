"""
HDF5 Demo File Inspector
Usage: python check_hdf5.py demos/demo_0001.hdf5
"""
import sys
import h5py
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python check_hdf5.py <path_to_hdf5>")
    sys.exit(1)

path = sys.argv[1]
f = h5py.File(path, "r")

# === Metadata ===
print("=" * 50)
print(f"📁 File: {path}")
print(f"   Duration: {f.attrs['duration_sec']:.2f}s")
print(f"   Frames: {f.attrs['num_frames']}")
print(f"   Images: {f.attrs['num_images']}")
print(f"   Created: {f.attrs['created_at']}")
print(f"   Subsample: {f.attrs['image_subsample']}")
print()

# === Dataset Shapes ===
print("📊 Datasets:")
for key in f.keys():
    print(f"   {key}: {f[key].shape} ({f[key].dtype})")
print()

# === Joint Angle Stats ===
joints = f["joint_angles"][:]
print("🦾 Joint Angle Stats (min / max):")
for i in range(joints.shape[1]):
    print(f"   joint_{i:2d}: [{joints[:, i].min():.3f}, {joints[:, i].max():.3f}]")
print()

# === Image FPS ===
if "image_timestamps" in f:
    img_ts = f["image_timestamps"][:]
    if len(img_ts) > 1:
        img_fps = (len(img_ts) - 1) / (img_ts[-1] - img_ts[0])
        print(f"📷 Image FPS: {img_fps:.1f}")

# === Show Sample Images ===
if "images" in f and f["images"].shape[0] > 0:
    try:
        import cv2
        images = f["images"]
        num_imgs = images.shape[0]
        # Show 5 evenly spaced frames
        sample_indices = np.linspace(0, num_imgs - 1, min(5, num_imgs), dtype=int)
        
        print(f"\n🖼 Showing {len(sample_indices)} sample images (press any key to advance, Q to quit)")
        for idx in sample_indices:
            img_rgb = images[idx]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            ts = f["image_timestamps"][idx]
            cv2.putText(img_bgr, f"Frame {idx}/{num_imgs} | t={ts:.2f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Demo Image", img_bgr)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
    except ImportError:
        print("⚠ OpenCV not available for image display")

f.close()
