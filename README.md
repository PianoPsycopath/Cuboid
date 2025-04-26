# Overview
Turns a rigged model into a cuboid based on existing bones and their closest vertices.
Optionally increase cuboid count for specific bones for higher detail

![image](https://github.com/user-attachments/assets/f073e3a3-4bfb-49b1-b241-12035f49323f)
![image](https://github.com/user-attachments/assets/1233ac74-a570-4242-95b0-4279b7bc5af6)

# Setup
1. Download or Copy Main.py into your scripts in Blender
2. Adjust Special Bones, Rotation Bones, Symmetrical Bone Pairs
   -Special Bones create higher cuboid counts
   -Rotation Bones increase possible rotation values to create the smallest possible cuboid within vertex bounds (UNUSED)
   -Symmetrical Bones insures bones remain the same size and orientation if they should be symmetrical
3. Select Armature, including bones and mesh
4. Run

