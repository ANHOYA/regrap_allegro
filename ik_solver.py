"""
Analytical FK + Jacobian IK Solver for Doosan A0509
Uses URDF kinematic chain parameters for accurate forward kinematics.
"""
import numpy as np


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _rpy_to_rot(r, p, y):
    return _rotz(y) @ _roty(p) @ _rotx(r)

def _make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class DoosanIK:
    """
    IK solver for Doosan A0509 6-DOF arm.
    Kinematic chain from URDF joint origins/axes.
    """
    def __init__(self):
        # Joint definitions from URDF: (xyz, rpy, axis)
        # All joints have axis (0,0,1) — rotation around local Z
        self.joints = [
            # joint_1: base → link_1
            {"xyz": [0, 0, 0.156], "rpy": [0, 0, 0]},
            # joint_2: link_1 → link_2
            {"xyz": [0, 0, 0], "rpy": [0, -1.571, -1.571]},
            # joint_3: link_2 → link_3
            {"xyz": [0.409, 0, 0], "rpy": [0, 0, 1.571]},
            # joint_4: link_3 → link_4
            {"xyz": [0, -0.367, 0], "rpy": [1.571, 0, 0]},
            # joint_5: link_4 → link_5
            {"xyz": [0, 0, 0], "rpy": [-1.571, 0, 0]},
            # joint_6: link_5 → link_6
            {"xyz": [0, -0.124, 0], "rpy": [1.571, 0, 0]},
        ]
        
        # Joint limits (all ±2.617 rad for A0509)
        self.joint_limits = np.array([[-2.617, 2.617]] * 6)
        
        # Default "ready" pose: arm bent forward, hand pointing down toward table
        self.default_pose = np.array([0.0, 0.8, 1.0, 0.0, 0.8, 0.0])

    def forward_kinematics(self, q):
        """
        Compute FK: returns list of (position, z_axis) for each joint frame,
        and the end-effector position.
        q: array of 6 joint angles
        """
        T = np.eye(4)
        positions = [T[:3, 3].copy()]  # Base position
        z_axes = [T[:3, 2].copy()]     # Base z-axis
        
        for i, joint in enumerate(self.joints):
            xyz = joint["xyz"]
            rpy = joint["rpy"]
            
            # Joint origin transform
            R_origin = _rpy_to_rot(*rpy)
            T_origin = _make_T(R_origin, xyz)
            
            # Joint rotation (around local Z)
            R_joint = _rotz(q[i])
            T_joint = _make_T(R_joint, [0, 0, 0])
            
            T = T @ T_origin @ T_joint
            positions.append(T[:3, 3].copy())
            z_axes.append(T[:3, 2].copy())
        
        return positions, z_axes, T[:3, 3].copy()

    def jacobian(self, q):
        """
        Compute 3×6 geometric Jacobian (position only).
        J_i = z_i × (p_ee - p_i) for revolute joints.
        """
        positions, z_axes, p_ee = self.forward_kinematics(q)
        J = np.zeros((3, 6))
        for i in range(6):
            J[:, i] = np.cross(z_axes[i], p_ee - positions[i])
        return J

    def solve_ik(self, q_current, target_pos, max_iter=5, gain=1.0, damping=0.01):
        """
        Damped least-squares IK (position only).
        Returns updated joint angles.
        """
        q = q_current.copy().astype(np.float64)
        
        for _ in range(max_iter):
            _, _, p_ee = self.forward_kinematics(q)
            error = target_pos.astype(np.float64) - p_ee
            error_norm = np.linalg.norm(error)
            
            if error_norm < 0.002:  # 2mm convergence
                break
            
            # Clamp error for stability
            if error_norm > 0.05:
                error = error * (0.05 / error_norm)
            
            J = self.jacobian(q)
            
            # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} e
            JJT = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error) * gain
            
            q += dq
            
            # Clamp to joint limits
            for i in range(6):
                q[i] = np.clip(q[i], self.joint_limits[i, 0], self.joint_limits[i, 1])
        
        return q.astype(np.float32)
