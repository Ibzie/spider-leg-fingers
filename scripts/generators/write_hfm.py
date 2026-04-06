import numpy as np
import os


class HumanHand5Finger:
    def __init__(self):
        self.finger_configs = [
            {"name": "thumb", "pos": (-0.02, -0.04, 0), "ori_z": -60},
            {"name": "index", "pos": (0.03, -0.02, 0), "ori_z": -15},
            {"name": "middle", "pos": (0.035, 0, 0), "ori_z": 0},
            {"name": "ring", "pos": (0.03, 0.02, 0), "ori_z": 15},
            {"name": "pinky", "pos": (0.02, 0.04, 0), "ori_z": 30},
        ]
        self.link_radius = 0.005
        self.proximal = 0.040
        self.middle = 0.025
        self.distal = 0.020
            
    def generate_mujoco_model(self):
        lines = []
        lines.append('<mujoco model="human_hand_5finger">')
        lines.append('  <compiler angle="degree"/>')
        lines.append('  <option timestep="0.002" gravity="0 0 -9.81"/>')
        lines.append('  <default><joint damping="0.05"/><geom friction="0.8 0.01 0.01"/></default>')
        lines.append('  <worldbody>')
        lines.append('    <light diffuse="0.5 0.5 0.5" pos="0 0 3" dir="0 0 -1"/>')
        lines.append('    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>')
        lines.append('    <body name="palm" pos="0 0 0.1">')
        lines.append('      <geom name="palm_geom" type="box" size="0.04 0.06 0.01" rgba="0.7 0.6 0.5 1"/>')
        lines.append('      <freejoint name="palm_root"/>')
        
        for cfg in self.finger_configs:
            name = cfg["name"]
            pos = cfg["pos"]
            ori = cfg["ori_z"]
            
            lines.append(f'      <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" euler="0 0 {ori}">')
            lines.append(f'        <joint name="{name}_mcp_flex" type="hinge" axis="0 1 0" range="-10 90"/>')
            if name != "thumb":
                lines.append(f'        <joint name="{name}_mcp_abd" type="hinge" axis="0 0 1" range="-20 20"/>')
            lines.append(f'        <geom name="{name}_proximal" type="capsule" size="{self.link_radius}" fromto="0 0 0 0 0 {self.proximal}" rgba="0.7 0.6 0.5 1"/>')
            lines.append(f'        <body name="{name}_mid" pos="0 0 {self.proximal}">')
            lines.append(f'          <joint name="{name}_pip" type="hinge" axis="0 1 0" range="0 100"/>')
            lines.append(f'          <geom name="{name}_middle" type="capsule" size="{self.link_radius*0.9}" fromto="0 0 0 0 0 {self.middle}" rgba="0.65 0.55 0.45 1"/>')
            lines.append(f'          <body name="{name}_dis" pos="0 0 {self.middle}">')
            lines.append(f'            <joint name="{name}_dip" type="hinge" axis="0 1 0" range="0 80"/>')
            lines.append(f'            <geom name="{name}_distal" type="capsule" size="{self.link_radius*0.8}" fromto="0 0 0 0 0 {self.distal}" rgba="0.6 0.5 0.4 1"/>')
            lines.append(f'            <site name="{name}_tip" pos="0 0 {self.distal}" size="0.005" type="sphere" rgba="1 0 0 0.5"/>')
            lines.append('          </body>')
            lines.append('        </body>')
            lines.append('      </body>')
        
        lines.append('    </body>')
        lines.append('  </worldbody>')
        lines.append('  <actuator>')
        for cfg in self.finger_configs:
            name = cfg["name"]
            lines.append(f'    <position name="{name}_mcp_flex_act" joint="{name}_mcp_flex" kp="80" kv="8"/>')
            if name != "thumb":
                lines.append(f'    <position name="{name}_mcp_abd_act" joint="{name}_mcp_abd" kp="40" kv="4"/>')
            lines.append(f'    <position name="{name}_pip_act" joint="{name}_pip" kp="60" kv="6"/>')
            lines.append(f'    <position name="{name}_dip_act" joint="{name}_dip" kp="50" kv="5"/>')
        lines.append('  </actuator>')
        lines.append('  <sensor>')
        for cfg in self.finger_configs:
            lines.append(f'    <touch name="{cfg["name"]}_touch" site="{cfg["name"]}_tip"/>')
        lines.append('  </sensor>')
        lines.append('</mujoco>')
        return chr(10).join(lines)


if __name__ == "__main__":
    hand = HumanHand5Finger()
    xml = hand.generate_mujoco_model()
    output_path = "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/assets/models/hfm_5finger.xml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml)
    print(f"Generated HFM model: {output_path}")