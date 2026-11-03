"""MJCF scene construction. For now a fixed probe scene; procedural parts follow."""


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
