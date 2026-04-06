import subprocess

spider_leg_code = '''
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class SpiderLegConfig:
    """Configuration for spider leg finger"""
    coxa_length: float = 0.015
    femur_length: float = 0.035
    tibia_length: float = 0.050
    coxa_range: Tuple[float, float] = (-np.pi/3, np.pi/3)
    femur_range: Tuple[float, float] = (-np.pi/4, np.pi/2)
    tibia_range: Tuple[float, float] = (0, 2*np.pi/3)
    fluid_bulk_modulus: float = 2.2e9
    link_radius: float = 0.004
    
    def __post_init__(self):
        self.total_length = self.coxa_length + self.femur_length + self.tibia_length


class SpiderLegKinematics:
    """Closed-form kinematics for spider leg finger"""
    
    def __init__(self, config: SpiderLegConfig = None):
        self.config = config or SpiderLegConfig()
        
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute end-effector position from joint angles"""
        t0, t1, t2 = joint_angles
        c = self.config
        
        x_coxa = c.coxa_length * np.cos(t0)
        y_coxa = c.coxa_length * np.sin(t0)
        z_coxa = 0.0
        
        x_femur = x_coxa + c.femur_length * np.cos(t0) * np.cos(t1)
        y_femur = y_coxa + c.femur_length * np.sin(t0) * np.cos(t1)
        z_femur = z_coxa + c.femur_length * np.sin(t1)
        
        t_tibia_abs = t1 - t2
        
        x = x_femur + c.tibia_length * np.cos(t0) * np.cos(t_tibia_abs)
        y = y_femur + c.tibia_length * np.sin(t0) * np.cos(t_tibia_abs)
        z = z_femur + c.tibia_length * np.sin(t_tibia_abs)
        
        return np.array([x, y, z])
    
    def inverse_kinematics(self, target_pos: np.ndarray, elbow_up: bool = True) -> np.ndarray:
        """Closed-form inverse kinematics"""
        x, y, z = target_pos
        c = self.config
        
        r = np.sqrt(x**2 + y**2)
        t0 = np.arctan2(y, x)
        
        r_eff = r - c.coxa_length
        d = np.sqrt(r_eff**2 + z**2)
        
        if d > c.femur_length + c.tibia_length:
            d = c.femur_length + c.tibia_length * 0.95
            
        cos_t2 = (c.femur_length**2 + c.tibia_length**2 - d**2) / (2 * c.femur_length * c.tibia_length)
        cos_t2 = np.clip(cos_t2, -1.0, 1.0)
        t2 = np.arccos(cos_t2)
        
        cos_t1_offset = (c.femur_length**2 + d**2 - c.tibia_length**2) / (2 * c.femur_length * d)
        cos_t1_offset = np.clip(cos_t1_offset, -1.0, 1.0)
        t1_offset = np.arccos(cos_t1_offset)
        
        t_base = np.arctan2(z, r_eff)
        
        if elbow_up:
            t1 = t_base + t1_offset
        else:
            t1 = t_base - t1_offset
            
        t2 = abs(t2)
        
        return np.array([t0, t1, t2])


class SpiderLegFinger:
    """Complete spider leg finger model"""
    
    def __init__(self, config: SpiderLegConfig = None):
        self.config = config or SpiderLegConfig()
        self.kinematics = SpiderLegKinematics(self.config)
        self.joint_angles = np.zeros(3)
        
    def solve_ik(self, target_pos: np.ndarray) -> np.ndarray:
        """Solve IK and update state"""
        angles = self.kinematics.inverse_kinematics(target_pos)
        self.joint_angles = angles
        return angles
    
    def get_end_effector_pos(self) -> np.ndarray:
        """Get current end-effector position"""
        return self.kinematics.forward_kinematics(self.joint_angles)
'''

# Write the file
with open(
    "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src/models/spider_leg.py",
    "w",
) as f:
    f.write(spider_leg_code.strip())

print("Created spider_leg.py")
