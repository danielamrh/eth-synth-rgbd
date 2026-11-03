"""MJCF scene construction. For now fixed demo scenes; procedural parts follow."""
import numpy as np

from synthrgbd.render.mujoco_renderer import camera_xml_attrs

# D435 depth stream at 848x480, f = 424 px (configs/camera_d435.yaml)
D435_W, D435_H, D435_F = 848, 480, 424.0


def demo_scene_xml() -> str:
    """Two aluminium profiles and an L-bracket on a matte table, D435 looking down at ~0.6 m."""
    K = np.array([[D435_F, 0, (D435_W - 1) / 2], [0, D435_F, (D435_H - 1) / 2], [0, 0, 1]])
    return f"""
<mujoco>
  <visual><global offwidth="{D435_W}" offheight="{D435_H}"/></visual>
  <asset>
    <material name="alu" rgba=".75 .76 .78 1" specular="1" shininess="1"/>
    <material name="table" rgba=".35 .35 .38 1" specular=".05"/>
  </asset>
  <worldbody>
    <light pos="0 -0.3 1.5" dir="0 0.2 -1"/>
    <camera name="cam" pos="0 -0.32 0.52" xyaxes="1 0 0 0 0.85 0.53" {camera_xml_attrs(K, D435_W, D435_H)}/>
    <geom name="table" type="plane" size="1 1 .1" material="table"/>
    <body name="profile_a" pos="-0.06 0 0.02" euler="0 0 20"><geom type="box" size=".15 .02 .02" material="alu"/></body>
    <body name="profile_b" pos="0.08 0.12 0.02" euler="0 0 -35"><geom type="box" size=".12 .02 .02" material="alu"/></body>
    <body name="bracket" pos="0.12 -0.08 0" euler="0 0 50">
      <geom type="box" size=".04 .03 .003" pos="0 0 .003" material="alu"/>
      <geom type="box" size=".003 .03 .04" pos="-.037 0 .04" material="alu"/>
    </body>
  </worldbody>
</mujoco>"""


def probe_scene_xml(width: int = 640, height: int = 480) -> str:
    """One aluminium bar on a table, camera looking down at ~53 deg."""
    return f"""
<mujoco>
  <visual><global offwidth="{width}" offheight="{height}"/></visual>
  <asset>
    <material name="alu" rgba=".7 .7 .75 1" specular="1" shininess="1"/>
    <material name="table" rgba=".6 .6 .6 1" specular=".1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2"/>
    <camera name="cam" pos="0 -0.6 0.5" xyaxes="1 0 0 0 0.6 0.8"/>
    <geom name="table" type="plane" size="1 1 .1" material="table"/>
    <body pos="0 0 0.02"><geom name="profile" type="box" size=".1 .01 .01" material="alu"/></body>
  </worldbody>
</mujoco>"""
