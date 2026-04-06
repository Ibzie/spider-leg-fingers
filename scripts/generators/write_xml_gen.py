import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import os


@dataclass
class SpiderLegConfig:
    """Configuration for spider leg finger"""

    coxa_length: float = 0.015
    femur_length: float = 0.035
    tibia_length: float = 0.050
    coxa_range: Tuple[float, float] = (-60, 60)  # degrees
    femur_range: Tuple[float, float] = (-45, 90)
    tibia_range: Tuple[float, float] = (0, 120)
    link_radius: float = 0.004


class SpiderHand5Finger:
    """5-finger spider leg hand"""

    def __init__(self, config: SpiderLegConfig = None):
        self.config = config or SpiderLegConfig()

        self.finger_configs = [
            {"name": "thumb", "pos": (-0.02, -0.04, 0), "ori_z": -60},
            {"name": "index", "pos": (0.03, -0.02, 0), "ori_z": -15},
            {"name": "middle", "pos": (0.035, 0, 0), "ori_z": 0},
            {"name": "ring", "pos": (0.03, 0.02, 0), "ori_z": 15},
            {"name": "pinky", "pos": (0.02, 0.04, 0), "ori_z": 30},
        ]

    def generate_mujoco_model(self) -> str:
        """Generate complete MuJoCo XML model"""
        c = self.config

        lines = []
        lines.append('<mujoco model="spider_hand_5finger">')
        lines.append('  <compiler angle="degree"/>')
        lines.append(
            '  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>'
        )
        lines.append("  <default>")
        lines.append('    <joint damping="0.1"/>')
        lines.append('    <geom friction="0.8 0.01 0.01"/>')
        lines.append("  </default>")
        lines.append("  <worldbody>")
        lines.append('    <light diffuse="0.5 0.5 0.5" pos="0 0 3" dir="0 0 -1"/>')
        lines.append(
            '    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>'
        )
        lines.append('    <body name="palm" pos="0 0 0.1">')
        lines.append(
            '      <geom name="palm_geom" type="box" size="0.04 0.06 0.01" rgba="0.5 0.4 0.3 1"/>'
        )
        lines.append('      <freejoint name="palm_root"/>')

        # Add fingers
        for cfg in self.finger_configs:
            name = cfg["name"]
            pos = cfg["pos"]
            ori = cfg["ori_z"]

            # Finger body
            lines.append(
                f'      <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" euler="0 0 {ori}">'
            )

            # Coxa (yaw joint)
            lines.append(
                f'        <joint name="{name}_coxa_joint" type="hinge" axis="0 0 1" range="{c.coxa_range[0]} {c.coxa_range[1]}"/>'
            )
            lines.append(
                f'        <geom name="{name}_coxa_geom" type="capsule" size="{c.link_radius}" fromto="0 0 0 {c.coxa_length} 0 0" rgba="0.4 0.3 0.2 1"/>'
            )

            # Femur body
            lines.append(
                f'        <body name="{name}_femur" pos="{c.coxa_length} 0 0">'
            )
            lines.append(
                f'          <joint name="{name}_femur_joint" type="hinge" axis="0 1 0" range="{c.femur_range[0]} {c.femur_range[1]}"/>'
            )
            lines.append(
                f'          <geom name="{name}_femur_geom" type="capsule" size="{c.link_radius}" fromto="0 0 0 {c.femur_length} 0 0" rgba="0.4 0.3 0.2 1"/>'
            )

            # Tibia body
            lines.append(
                f'          <body name="{name}_tibia" pos="{c.femur_length} 0 0">'
            )
            lines.append(
                f'            <joint name="{name}_tibia_joint" type="hinge" axis="0 1 0" range="{c.tibia_range[0]} {c.tibia_range[1]}"/>'
            )
            lines.append(
                f'            <geom name="{name}_tibia_geom" type="capsule" size="{c.link_radius * 0.8}" fromto="0 0 0 {c.tibia_length} 0 0" rgba="0.3 0.2 0.15 1"/>'
            )
            lines.append(
                f'            <site name="{name}_tip" pos="{c.tibia_length} 0 0" size="0.005" type="sphere" rgba="1 0 0 0.5"/>'
            )
            lines.append("          </body>")  # end tibia
            lines.append("        </body>")  # end femur
            lines.append("      </body>")  # end finger

        lines.append("    </body>")  # end palm
        lines.append("  </worldbody>")

        # Actuators
        lines.append("  <actuator>")
        for cfg in self.finger_configs:
            name = cfg["name"]
            lines.append(
                f'    <position name="{name}_coxa_act" joint="{name}_coxa_joint" kp="100" kv="10"/>'
            )
            lines.append(
                f'    <position name="{name}_femur_act" joint="{name}_femur_joint" kp="100" kv="10"/>'
            )
            lines.append(
                f'    <position name="{name}_tibia_act" joint="{name}_tibia_joint" kp="150" kv="15"/>'
            )
        lines.append("  </actuator>")

        # Sensors
        lines.append("  <sensor>")
        for cfg in self.finger_configs:
            name = cfg["name"]
            lines.append(f'    <touch name="{name}_touch" site="{name}_tip"/>')
        lines.append("  </sensor>")

        lines.append("</mujoco>")

        return "\n".join(lines)


if __name__ == "__main__":
    hand = SpiderHand5Finger()
    xml = hand.generate_mujoco_model()

    # Save to file
    output_path = "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/assets/models/slfm_5finger.xml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml)

    print(f"Generated SLFM model: {output_path}")
    print(f"Model size: {len(xml)} characters")
