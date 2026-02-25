"""
Demonstration Data Recorder for Imitation Learning
Records joint angles + camera images to HDF5 files.
"""

import os
import time
import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")


class DemoRecorder:
    """Records demonstration data (joint angles + images) to HDF5."""

    def __init__(self, save_dir="demos", image_subsample=6, jpeg_quality=90):
        """
        Args:
            save_dir: Directory to save HDF5 files
            image_subsample: Save every N-th frame's image (6 = ~10fps from 60fps)
            jpeg_quality: JPEG compression quality (1-100)
        """
        self.save_dir = save_dir
        self.image_subsample = image_subsample
        self.jpeg_quality = jpeg_quality

        os.makedirs(save_dir, exist_ok=True)

        # Recording buffers
        self._joint_buf = []
        self._time_buf = []
        self._image_buf = []
        self._image_time_buf = []
        self._frame_count = 0
        self._start_time = None
        self._is_recording = False

    @property
    def is_recording(self):
        return self._is_recording

    @property
    def elapsed(self):
        """Elapsed time since recording started (seconds)."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def frame_count(self):
        return self._frame_count

    def start(self):
        """Start a new recording session. Clears all buffers."""
        self._joint_buf = []
        self._time_buf = []
        self._image_buf = []
        self._image_time_buf = []
        self._frame_count = 0
        self._start_time = time.time()
        self._is_recording = True

    def add_frame(self, joint_angles, image=None):
        """
        Add a frame to the recording buffer.

        Args:
            joint_angles: np.array (16,) float32 — current joint angles
            image: np.array (H, W, 3) uint8 — RGB camera frame (optional)
        """
        if not self._is_recording:
            return

        t = time.time() - self._start_time
        self._joint_buf.append(joint_angles.copy())
        self._time_buf.append(t)
        self._frame_count += 1

        # Subsample images to reduce file size
        if image is not None and (self._frame_count % self.image_subsample == 0):
            self._image_buf.append(image.copy())
            self._image_time_buf.append(t)

    def discard(self):
        """Discard current recording without saving."""
        self._joint_buf = []
        self._time_buf = []
        self._image_buf = []
        self._image_time_buf = []
        self._frame_count = 0
        self._start_time = None
        self._is_recording = False

    def save(self, joint_names=None):
        """
        Save current recording to HDF5 and clear buffers.

        Args:
            joint_names: list of joint name strings (optional metadata)

        Returns:
            str: Path to saved file, or None if nothing to save
        """
        if not self._joint_buf:
            self._is_recording = False
            return None

        path = self._get_next_path()
        joints = np.array(self._joint_buf, dtype=np.float32)   # (N, 16)
        times = np.array(self._time_buf, dtype=np.float64)      # (N,)

        with h5py.File(path, "w") as f:
            f.create_dataset("joint_angles", data=joints, compression="gzip", compression_opts=4)
            f.create_dataset("timestamps", data=times)

            if self._image_buf:
                images = np.array(self._image_buf, dtype=np.uint8)  # (M, H, W, 3)
                image_times = np.array(self._image_time_buf, dtype=np.float64)
                f.create_dataset("images", data=images, compression="gzip", compression_opts=4)
                f.create_dataset("image_timestamps", data=image_times)

            # Metadata
            f.attrs["num_frames"] = len(joints)
            f.attrs["num_images"] = len(self._image_buf)
            f.attrs["duration_sec"] = float(times[-1]) if len(times) > 0 else 0.0
            f.attrs["num_joints"] = joints.shape[1] if joints.ndim == 2 else 0
            f.attrs["image_subsample"] = self.image_subsample
            f.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if joint_names is not None:
                f.attrs["joint_names"] = list(joint_names)

        # Reset
        self.discard()
        return path

    def _get_next_path(self):
        """Auto-number: demo_0001.hdf5, demo_0002.hdf5, ..."""
        idx = 1
        while True:
            path = os.path.join(self.save_dir, f"demo_{idx:04d}.hdf5")
            if not os.path.exists(path):
                return path
            idx += 1

    def status_str(self):
        """Return a status string for display."""
        if not self._is_recording:
            return "[IDLE] Press S=Start"
        elapsed = self.elapsed
        mins, secs = divmod(elapsed, 60)
        return f"[REC ●] {int(mins):02d}:{secs:05.2f} | {self._frame_count} frames | {len(self._image_buf)} imgs"
