# Ballpark

Given a 3D mesh or a robot URDF, create a "ballpark" estimate of its spherical collision geometry.

![Sphere decompositions for various robots](assets/splash.png)

Features include:
- Fast mesh-to-sphere decomposition via recursive PCA-based splitting.
- 使用与 cuRobo v2 MorphIt 对齐的覆盖、外凸、相切、重叠和可选半平面损失逐 link 细化球体。
- Different presets for conservative, balanced, or surface-fitting sphere sets.

For robot URDFs, we also include:
- Automatic sphere distribution across robot links, proportional to their geometry complexity.
- 每个 link 的球体独立细化，避免跨 link 的尺度耦合。
- Similar links are detected and share sphere parameters for visual and geometric consistency.
- JSON export with sphere parameters for each link, and an ignore-list of link pairs for collision checking.

We also include a set of interactive visualization tools, supported via [viser](https://viser.studio).


## Installation

```bash
pip install -e .
pip install -e ".[examples]"
pip install -e ".[dev]"  # with development tools (linting, testing)
```

## Quick Start

### Mesh Spherization

```python
import trimesh
from ballpark import spherize

# Load mesh
mesh = trimesh.load("object.stl")

# Generate spheres with adaptive fitting
spheres = spherize(mesh, target_spheres=32)

for s in spheres:
    print(f"center={s.center}, radius={s.radius}")
```

### Robot URDF Spherization

```python
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from ballpark import Robot, BallparkConfig, SpherePreset

# Load robot URDF with collision meshes
urdf = load_robot_description("panda_description")
urdf_coll = yourdfpy.URDF(
    robot=urdf.robot,
    load_collision_meshes=True,
)

# Create robot and generate spheres
robot = Robot(urdf_coll)
result = robot.spherize(target_spheres=100)

# 可选：使用固定球数的 MorphIt 五项代价逐 link 细化
config = BallparkConfig.from_preset(SpherePreset.BALANCED)
result = robot.refine(result, config=config)

for link_name, spheres in result.link_spheres.items():
    print(f"{link_name}: {len(spheres)} spheres")
```

### Manual Sphere Adjustments

We include an interactive script for adjusting the pose + radii of the auto-generated spheres. 

https://github.com/user-attachments/assets/895bbf5f-e4db-47c2-8946-cb5f2dbbb9b9


## Acknowledgments

This project builds on ideas from:
- [foam](https://github.com/CoMMALab/foam)
- [MorphIt](https://github.com/HIRO-group/MorphIt-1)
