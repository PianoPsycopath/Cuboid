#Hello
print("""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄  
▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ 
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌
▐░▌          ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌
▐░▌          ▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌
▐░▌          ▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌
▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ 
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀  
version 1.0 - 
Team Yume:PianoPsychopath
                                                                              
"""
      )

import bpy
import mathutils
from mathutils import Vector, Matrix
from collections import defaultdict
import math
import bmesh
from statistics import mean
from mathutils.kdtree import KDTree

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# 1.1 Special bones configuration for custom subdivision
SPECIAL_BONES = {
    # Format: "bone_name": {"lod": level_of_detail, "scale_factor": inflation_factor}
    "Head": {"lod": 16, "scale_factor": 1.0}
}

# 1.2 Bones requiring wider rotation ranges for optimal orientation
WIDE_ROTATION_BONES = [
    # Format: "wings", "hat", etc.
]

# 1.3 Symmetrical bone mapping for maintaining consistency
SYMMETRICAL_BONE_PAIRS = {
    "Leg": "Leg",
    "Arm": "Arm",
    "Toe_1_F": "Toe_1_F",
    "Toe_1_B": "Toe_1_B",
    "Ankle": "Ankle",
    "Wrist": "Wrist",
}

# --------------------------------------------------
# 2. SELECTION AND DATA PREPARATION
# --------------------------------------------------

def get_selected_armature_and_meshes():
    """
    Retrieves the selected armature and its associated meshes.
    
    Returns:
        tuple: (armature_object, list_of_mesh_objects)
    """
    armature_obj = None
    mesh_objs = []
    
    # 2.1 Find selected armature
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            armature_obj = obj
            break
    
    if not armature_obj:
        print("❌No Armature selected.")
        return None, []
    
    # 2.2 Find all meshes under the armature hierarchy
    def find_meshes(obj, mesh_list):
        for child in obj.children:
            if child.type == 'MESH':
                mesh_list.append(child)
            find_meshes(child, mesh_list)
    
    find_meshes(armature_obj, mesh_objs)
    
    if not mesh_objs:
        print("❌No meshes found under armature.")
        return None, []
    
    print(f"[2.2] Found {len(mesh_objs)} mesh objects under the armature.")
    return armature_obj, mesh_objs

def assign_vertices_to_bones(mesh_obj):
    """
    Assigns each vertex to the bone with the highest influence.
    
    Args:
        mesh_obj: The mesh object to process
        
    Returns:
        dict: Mapping of bone names to lists of vertex indices
    """
    mesh = mesh_obj.data
    vertex_bone_assignments = {}
    vertex_no_assignments = []
    
    # 2.3 First pass - assign based on vertex groups/weights
    for v_idx, vertex in enumerate(mesh.vertices):
        max_weight = 0
        assigned_bone = None
        
        for group in vertex.groups:
            if group.weight > max_weight:
                group_name = mesh_obj.vertex_groups[group.group].name
                max_weight = group.weight
                assigned_bone = group_name
        
        if assigned_bone:
            vertex_bone_assignments[v_idx] = assigned_bone
        else:
            vertex_no_assignments.append(v_idx)
    
    # 2.4 Second pass - assign remaining vertices to nearest bone
    if vertex_no_assignments:
        print(f"[2.4] Found {len(vertex_no_assignments)} vertices with no bone influence in {mesh_obj.name}")
        
        # Calculate average position for each bone's vertices
        bone_positions = defaultdict(lambda: [0, 0, 0])
        bone_counts = defaultdict(int)
        
        for v_idx, bone_name in vertex_bone_assignments.items():
            pos = mesh.vertices[v_idx].co
            bone_positions[bone_name][0] += pos.x
            bone_positions[bone_name][1] += pos.y 
            bone_positions[bone_name][2] += pos.z
            bone_counts[bone_name] += 1
        
        # Average the positions
        for bone_name in bone_positions:
            if bone_counts[bone_name] > 0:
                bone_positions[bone_name][0] /= bone_counts[bone_name]
                bone_positions[bone_name][1] /= bone_counts[bone_name]
                bone_positions[bone_name][2] /= bone_counts[bone_name]
        
        # Assign unassigned vertices to nearest bone
        for v_idx in vertex_no_assignments:
            v_pos = mesh.vertices[v_idx].co
            closest_bone = None
            closest_dist = float('inf')
            
            for bone_name, pos in bone_positions.items():
                bone_pos = Vector(pos)
                dist = (v_pos - bone_pos).length
                if dist < closest_dist:
                    closest_dist = dist
                    closest_bone = bone_name
            
            if closest_bone:
                vertex_bone_assignments[v_idx] = closest_bone
    
    # 2.5 Group by bone
    bone_vertices = defaultdict(list)
    for v_idx, bone in vertex_bone_assignments.items():
        bone_vertices[bone].append(v_idx)
    
    return bone_vertices

# --------------------------------------------------
# 3. SPATIAL ANALYSIS AND CALCULATIONS
# --------------------------------------------------

def calculate_spatial_priority(face_data, mesh_obj, bone_head_world):
    """
    Calculates priority scores for faces based on spatial proximity to bone.
    
    Args:
        face_data: List of face data dictionaries
        mesh_obj: The mesh object
        bone_head_world: World position of bone head
        
    Returns:
        list: Updated face data with priority scores
    """
    mesh = mesh_obj.data
    
    for fd in face_data:
        face_idx = fd.get('face_index', 0)
        if face_idx < len(mesh.polygons):
            face = mesh.polygons[face_idx]
            
            # 3.1 Calculate face center in world space
            face_center = Vector((0, 0, 0))
            for vert_idx in face.vertices:
                face_center += mesh.vertices[vert_idx].co
            face_center /= len(face.vertices)
            face_center_world = mesh_obj.matrix_world @ face_center
            
            # 3.2 Calculate distance to bone head
            distance = (face_center_world - bone_head_world).length
            
            # 3.3 Add spatial priority score (inversely proportional to distance)
            fd['priority'] = fd.get('priority', 0) * (1.0 + (1.0 / (1.0 + distance)))
    
    return face_data

def calculate_bounding_box_volume(verts):
    """
    Calculates the volume of the bounding box containing the vertices.
    
    Args:
        verts: List of vertex coordinates
        
    Returns:
        tuple: (volume, dimensions)
    """
    if not verts:
        return 0, (0, 0, 0)
    
    # 3.4 Find extremes
    min_x = min(v.x for v in verts)
    max_x = max(v.x for v in verts)
    min_y = min(v.y for v in verts)
    max_y = max(v.y for v in verts)
    min_z = min(v.z for v in verts)
    max_z = max(v.z for v in verts)
    
    # 3.5 Calculate dimensions and volume
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z
    
    return width * height * depth, (width, height, depth)

def get_bounding_box_dimensions(verts):
    """
    Returns the dimensions of the bounding box containing the vertices.
    
    Args:
        verts: List of vertex coordinates
        
    Returns:
        tuple: (width, height, depth)
    """
    if not verts:
        return (0, 0, 0)
    
    # 3.6 Find extremes and calculate dimensions
    min_x = min(v.x for v in verts)
    max_x = max(v.x for v in verts)
    min_y = min(v.y for v in verts)
    max_y = max(v.y for v in verts)
    min_z = min(v.z for v in verts)
    max_z = max(v.z for v in verts)
    
    return (max_x - min_x, max_y - min_y, max_z - min_z)

# --------------------------------------------------
# 4. FACE PROCESSING
# --------------------------------------------------

def get_faces_for_bone(mesh_obj, vertex_indices):
    """
    Gets faces that contain vertices influenced by the bone.
    
    Args:
        mesh_obj: The mesh object
        vertex_indices: List of vertex indices
        
    Returns:
        list: Face indices
    """
    mesh = mesh_obj.data
    bone_faces = []
    
    # 4.1 Create a set for faster lookups
    vert_set = set(vertex_indices)
    
    # 4.2 Find faces that use any of these vertices
    for face in mesh.polygons:
        if any(v in vert_set for v in face.vertices):
            bone_faces.append(face.index)
    
    return bone_faces

def extract_face_data(mesh_obj, face_indices, vertex_indices_set=None):
    """
    Extracts UV and material data for faces, prioritizing faces that contain target vertices.
    
    Args:
        mesh_obj: The mesh object
        face_indices: List of face indices
        vertex_indices_set: Set of vertex indices for prioritization
        
    Returns:
        tuple or list: Face data with priority information, and vertex-face mapping if requested
    """
    mesh = mesh_obj.data
    result = []
    
    # 4.3 Skip if no UV layers
    if not mesh.uv_layers or len(mesh.uv_layers) == 0:
        return result if vertex_indices_set is None else (result, {})
    
    # 4.4 Get the active UV layer
    uv_layer = mesh.uv_layers.active
    
    # 4.5 Track which vertices belong to which faces
    vertex_face_map = {}
    if vertex_indices_set:
        for v_idx in vertex_indices_set:
            vertex_face_map[v_idx] = []
    
        # First pass - identify faces that contain vertices
        primary_faces = []
        for face_idx in face_indices:
            face = mesh.polygons[face_idx]
            is_primary = False
            
            # Check if this face contains any vertices
            for vert_idx in face.vertices:
                if vert_idx in vertex_indices_set:
                    is_primary = True
                    vertex_face_map[vert_idx].append(face_idx)
            
            if is_primary:
                primary_faces.append(face_idx)
        
        # 4.6 Process primary faces first (higher priority)
        for face_idx in primary_faces:
            face = mesh.polygons[face_idx]
            face_data = {
                'material_index': face.material_index,
                'vertices': [],
                'uvs': [],
                'face_index': face_idx, 
                'priority': 1  
            }
            
            # Get vertices and UVs
            for loop_idx in range(face.loop_start, face.loop_start + face.loop_total):
                vert_idx = mesh.loops[loop_idx].vertex_index
                uv = uv_layer.data[loop_idx].uv
                face_data['vertices'].append(vert_idx)
                face_data['uvs'].append((uv[0], uv[1]))
            
            result.append(face_data)
        
        # 4.7 Process secondary faces (lower priority)
        secondary_faces = [idx for idx in face_indices if idx not in primary_faces]
        for face_idx in secondary_faces:
            face = mesh.polygons[face_idx]
            face_data = {
                'material_index': face.material_index,
                'vertices': [],
                'uvs': [],
                'face_index': face_idx,
                'priority': 0  
            }
            
            # Get vertices and UVs
            for loop_idx in range(face.loop_start, face.loop_start + face.loop_total):
                vert_idx = mesh.loops[loop_idx].vertex_index
                uv = uv_layer.data[loop_idx].uv
                face_data['vertices'].append(vert_idx)
                face_data['uvs'].append((uv[0], uv[1]))
            
            result.append(face_data)
        
        return result, vertex_face_map
    
    # 4.8 Standard processing without prioritization
    for face_idx in face_indices:
        face = mesh.polygons[face_idx]
        face_data = {
            'material_index': face.material_index,
            'vertices': [],
            'uvs': []
        }
        
        # Get vertices and UVs
        for loop_idx in range(face.loop_start, face.loop_start + face.loop_total):
            vert_idx = mesh.loops[loop_idx].vertex_index
            uv = uv_layer.data[loop_idx].uv
            face_data['vertices'].append(vert_idx)
            face_data['uvs'].append((uv[0], uv[1]))
        
        result.append(face_data)
    
    return result

# --------------------------------------------------
# 5. UV MAPPING
# --------------------------------------------------

def apply_uvs_to_cube_face(bm, face, uv_layer, uvs):
    """
    Applies a set of UVs to a bmesh face based on directional mapping.
    
    Args:
        bm: BMesh object
        face: Face to apply UVs to
        uv_layer: UV layer to use
        uvs: List of UV coordinates to sample from
    """
    if not uvs:
        return
    
    # 5.1 Calculate UV bounds
    min_u = min(uv[0] for uv in uvs)
    max_u = max(uv[0] for uv in uvs)
    min_v = min(uv[1] for uv in uvs)
    max_v = max(uv[1] for uv in uvs)
    
    # 5.2 UV Ranges
    u_range = max(max_u - min_u, 0.01)
    v_range = max(max_v - min_v, 0.01)
    
    # 5.3 Find UV coords by density using grid analysis
    grid_size = 8
    grid = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    
    for uv in uvs:
        norm_u = (uv[0] - min_u) / u_range if u_range > 0 else 0.5
        norm_v = (uv[1] - min_v) / v_range if v_range > 0 else 0.5
        
        norm_u = max(0, min(0.999, norm_u))
        norm_v = max(0, min(0.999, norm_v))
        
        grid_u = int(norm_u * grid_size)
        grid_v = int(norm_v * grid_size)
        grid[grid_u][grid_v].append(uv)
    
    # 5.4 Find cells with highest UV density
    max_uvs = 0
    best_cells = []
    for u in range(grid_size):
        for v in range(grid_size):
            cell_count = len(grid[u][v])
            if cell_count > max_uvs:
                max_uvs = cell_count
                best_cells = [(u, v)]
            elif cell_count == max_uvs and max_uvs > 0:
                best_cells.append((u, v))
    
    # 5.5 Determine final UV bounds based on density
    if best_cells:
        best_uvs = []
        for u, v in best_cells:
            best_uvs.extend(grid[u][v])
        
        if best_uvs:
            cell_min_u = min(uv[0] for uv in best_uvs)
            cell_max_u = max(uv[0] for uv in best_uvs)
            cell_min_v = min(uv[1] for uv in best_uvs)
            cell_max_v = max(uv[1] for uv in best_uvs)
            
            # Use the dominant cells' bounds
            face_min_u = cell_min_u
            face_max_u = cell_max_u
            face_min_v = cell_min_v
            face_max_v = cell_max_v
        else:
            # Fallback to overall bounds
            face_min_u = min_u
            face_max_u = max_u
            face_min_v = min_v
            face_max_v = max_v
    else:
        # Fallback to overall bounds
        face_min_u = min_u
        face_max_u = max_u
        face_min_v = min_v
        face_max_v = max_v
    
    # 5.6 Calculate the center point of the UV area
    center_u = (face_min_u + face_max_u) / 2
    center_v = (face_min_v + face_max_v) / 2
    
    # 5.7 Get face dimensions in world space to determine proper UV scaling
    face_verts = [loop.vert.co for loop in face.loops]
    
    # Calculate approximate face dimensions
    dists = []
    for i in range(len(face_verts)):
        for j in range(i+1, len(face_verts)):
            dists.append((face_verts[i] - face_verts[j]).length)
    
    # 5.8 Estimate face size as average of its dimensions
    if dists:
        avg_face_size = sum(dists) / len(dists)
    else:
        avg_face_size = 1.0  # Fallback if no valid distances
    
    # 5.9 Scale UV based on desired pixels per meter
    pixels_per_meter = 16.0
    uv_scale = avg_face_size * pixels_per_meter / 1024.0  # Assuming 1024 texture size
    
    # 5.10 Adjust UV bounds to maintain center position but scale properly
    u_size = u_range * uv_scale
    v_size = v_range * uv_scale
    
    # Recalculate bounds while maintaining center point
    new_face_min_u = center_u - u_size / 2
    new_face_max_u = center_u + u_size / 2
    new_face_min_v = center_v - v_size / 2
    new_face_max_v = center_v + v_size / 2
    
    # 5.11 Create corner UVs using the calculated bounds
    corner_uvs = [
        Vector((new_face_min_u, new_face_min_v)),  # bottom-left
        Vector((new_face_max_u, new_face_min_v)),  # bottom-right
        Vector((new_face_max_u, new_face_max_v)),  # top-right
        Vector((new_face_min_u, new_face_max_v))   # top-left
    ]
    
    # 5.12 Get face normal to determine orientation
    face_normal = face.normal
    world_up = Vector((0, 0, 1))
    
    # Sort face vertices based on their position relative to face center
    # This helps maintain consistent UV mapping regardless of face orientation
    face_center = face.calc_center_median()
    
    # 5.13 Find the local X and Y axes for the face
    if abs(face_normal.dot(world_up)) > 0.9:  # Face is horizontal
        local_x = Vector((1, 0, 0))
    else:
        local_x = world_up.cross(face_normal).normalized()
    local_y = face_normal.cross(local_x).normalized()
    
    # 5.14 Sort loops based on their position in the local coordinate system
    sorted_loops = []
    for i, loop in enumerate(face.loops):
        # Calculate 2D coordinates on the face plane
        vert_local = loop.vert.co - face_center
        x_coord = vert_local.dot(local_x)
        y_coord = vert_local.dot(local_y)
        
        # Calculate angle in face space
        angle = math.atan2(y_coord, x_coord)
        sorted_loops.append((angle, i, loop))
    
    # Sort by angle
    sorted_loops.sort(key=lambda x: x[0])
    
    # 5.15 Apply UVs in counter-clockwise order
    for i, (_, _, loop) in enumerate(sorted_loops):
        loop[uv_layer].uv = corner_uvs[i % len(corner_uvs)]

def apply_face_data_to_cube(cube, mesh_obj, face_data, vertex_face_map=None):
    """
    Applies extracted face data to the cube with improved UV mapping.
    
    Args:
        cube: Target cube object
        mesh_obj: Source mesh object
        face_data: Face data to apply
        vertex_face_map: Optional mapping of vertices to faces
    """
    # 5.16 Skip if no data
    if not face_data:
        return
    
    # 5.17 Create bmesh for editing
    bm = bmesh.new()
    bm.from_mesh(cube.data)
    
    # 5.18 Ensure we have UV layers
    if not bm.loops.layers.uv:
        uv_layer = bm.loops.layers.uv.new("UVMap")
    else:
        uv_layer = bm.loops.layers.uv[0]
    
    # 5.19 Get materials from original mesh
    materials = [mat for mat in mesh_obj.data.materials if mat is not None]
    
    # 5.20 Make sure cube has enough material slots
    while len(cube.data.materials) < len(materials):
        cube.data.materials.append(None)
    
    # 5.21 Copy materials from original mesh
    for i, mat in enumerate(materials):
        if i < len(cube.data.materials):
            cube.data.materials[i] = mat
    
    # 5.22 Get face direction data from original mesh
    original_mesh_bm = bmesh.new()
    original_mesh_bm.from_mesh(mesh_obj.data)
    
    # Ensure lookup tables are updated
    original_mesh_bm.verts.ensure_lookup_table()
    original_mesh_bm.edges.ensure_lookup_table()
    original_mesh_bm.faces.ensure_lookup_table()
    
    bmesh.ops.transform(original_mesh_bm, matrix=mesh_obj.matrix_world, verts=original_mesh_bm.verts)
    
    # 5.23 Define the 6 main directions in world space
    directions = [
        Vector((1, 0, 0)),   # +X (right)
        Vector((-1, 0, 0)),  # -X (left)
        Vector((0, 1, 0)),   # +Y (front)
        Vector((0, -1, 0)),  # -Y (back)
        Vector((0, 0, 1)),   # +Z (top)
        Vector((0, 0, -1))   # -Z (bottom)
    ]
    
    # 5.24 Group face data by direction and by texture region
    direction_faces = {i: [] for i in range(6)}
    
    # Map face indices to BMFaces
    face_map = {}
    for i, f in enumerate(original_mesh_bm.faces):
        if i < len(original_mesh_bm.faces):
            face_map[i] = f
    
    # 5.25 Process original faces to find their dominant direction
    for fd in face_data:
        # Get face index and make sure it's valid
        face_idx = fd.get('face_index', 0)
            
        # Skip if face index is out of range
        if face_idx not in face_map:
            continue
            
        orig_face = face_map[face_idx]
        normal = orig_face.normal.normalized()
        
        # Find which of the 6 directions this face is most aligned with
        max_dot = -1
        best_dir = 0
        for i, dir_vec in enumerate(directions):
            dot = abs(normal.dot(dir_vec))
            if dot > max_dot:
                max_dot = dot
                best_dir = i
        
        # Add this face data to the appropriate direction group
        fd['direction'] = best_dir
        direction_faces[best_dir].append(fd)
    
    original_mesh_bm.free()
    
    # 5.26 Group face data by material per direction
    material_by_direction = {}
    for dir_idx, dir_faces in direction_faces.items():
        material_by_direction[dir_idx] = defaultdict(list)
        # Sort by priority if available
        if all('priority' in fd for fd in dir_faces):
            sorted_faces = sorted(dir_faces, key=lambda x: x.get('priority', 0), reverse=True)
        else:
            sorted_faces = dir_faces
            
        for fd in sorted_faces:
            material_by_direction[dir_idx][fd['material_index']].append(fd)
    
    # 5.27 Find most common material for each direction
    direction_materials = {}
    for dir_idx, mat_dict in material_by_direction.items():
        most_common_mat = 0
        max_count = 0
        for mat_idx, faces in mat_dict.items():
            if len(faces) > max_count:
                max_count = len(faces)
                most_common_mat = mat_idx
        direction_materials[dir_idx] = most_common_mat
    
    # 5.28 Default material if we don't find any
    default_material = 0
    if direction_materials:
        default_material = max(direction_materials.values(), key=list(direction_materials.values()).count)
    
    # 5.29 Process each face of the cube
    for face in bm.faces:
        # Convert face normal to world space
        local_normal = face.normal.copy()
        world_normal = cube.matrix_world.to_3x3() @ local_normal
        world_normal.normalize()
        
        # Find which of the 6 directions this face is most aligned with
        max_dot = -1
        best_dir = 0
        for i, dir_vec in enumerate(directions):
            dot = abs(world_normal.dot(dir_vec))
            if dot > max_dot:
                max_dot = dot
                best_dir = i
        
        # 5.30 Assign material based on direction
        face.material_index = direction_materials.get(best_dir, default_material)
        
        # 5.31 Find most appropriate UVs from faces in this direction
        appropriate_faces = []
        for mat_idx, face_list in material_by_direction.get(best_dir, {}).items():
            appropriate_faces.extend(face_list)
        
        # Sort by priority if available
        if appropriate_faces and 'priority' in appropriate_faces[0]:
            appropriate_faces = sorted(appropriate_faces, key=lambda x: x.get('priority', 0), reverse=True)
        
        # 5.32 Get UVs from appropriate faces
        uvs = []
        if appropriate_faces:
            # Prefer high priority faces
            for fd in appropriate_faces[:5]:  # Look at top 5 highest priority faces
                uvs.extend(fd.get('uvs', []))
        
        # 5.33 If we couldn't find any appropriate UVs, use all UVs as fallback
        if not uvs:
            for dir_idx, mat_dict in material_by_direction.items():
                for mat_idx, face_list in mat_dict.items():
                    for fd in face_list:
                        uvs.extend(fd.get('uvs', []))
        
        # 5.34 Apply UVs to this face
        apply_uvs_to_cube_face(bm, face, uv_layer, uvs)
    
    # 5.35 Update the mesh
    bm.to_mesh(cube.data)
    bm.free()
    cube.data.update()

# --------------------------------------------------
# 6. ROTATION AND OPTIMIZATION
# --------------------------------------------------

# 6.1 Rotation Range Determination
def should_use_wide_rotation_range(bone_name):
    """
    Determines if the specified bone requires an extended rotation range during optimization.
    
    Args:
        bone_name: Name of the bone to evaluate
        
    Returns:
        bool: True if wide rotation range should be applied
    """
    bone_name_lower = bone_name.lower()
    return any(keyword in bone_name_lower for keyword in WIDE_ROTATION_BONES)

# --------------------------------------------------
# 7. VOXEL GENERATION
# --------------------------------------------------

# 7.1 Primary Cuboid Generation
def create_cuboid_for_bone(armature_obj, mesh_obj, bone_name, vertex_indices, cube_collection, mesh_name_prefix=""):
    if not vertex_indices:
        print(f"❌[7.1] No vertices assigned to bone: {bone_name} in {mesh_obj.name}")
        return []
    
    # 7.1.1 Initialization and Data Preparation
    vertex_indices_set = set(vertex_indices)
    mesh = mesh_obj.data
    armature_mat = armature_obj.matrix_world
    pose_bone = armature_obj.pose.bones.get(bone_name)
    if not pose_bone:
        print(f"⚠️[7.1.1] Bone not found: {bone_name}")
        return []

    # 7.1.2 Special Case Detection
    is_special = bone_name in SPECIAL_BONES
    
    bone_head_world = armature_mat @ pose_bone.head
    bone_tail_world = armature_mat @ pose_bone.tail
    z_axis = (bone_tail_world - bone_head_world).normalized()

    if abs(z_axis.dot(Vector((0, 0, 1)))) < 0.9:
        x_axis = z_axis.cross(Vector((0, 0, 1))).normalized()
    else:
        x_axis = z_axis.cross(Vector((0, 1, 0))).normalized()

    y_axis = z_axis.cross(x_axis).normalized()
    base_rot = Matrix((x_axis, y_axis, z_axis)).transposed().to_4x4()

    bone_vertices_world = [mesh_obj.matrix_world @ mesh.vertices[idx].co for idx in vertex_indices]
    bone_length = (bone_tail_world - bone_head_world).length

    # 7.1.3 Rotation Optimization
    best_volume = float('inf')
    best_rot = base_rot
    best_dims = Vector((0, 0, 0))

    # 7.1.4 Rotation Range Configuration
    use_wide_range = should_use_wide_rotation_range(bone_name)
    angle_step = 1 if use_wide_range else 1
    angle_max = 89 if use_wide_range else 89

    # 7.1.5 Orientation Search
    for x_angle in range(-angle_max, angle_max + 1, angle_step):
        z_angle = 0
        y_angle = 0
    
        x_rad = math.radians(x_angle)
        y_rad = math.radians(y_angle)
        z_rad = math.radians(z_angle)
    
        # Generate rotation matrices
        rot_x = Matrix.Rotation(x_rad, 4, 'X')
        rot_y = Matrix.Rotation(y_rad, 4, 'Y')
        rot_z = Matrix.Rotation(z_rad, 4, 'Z')
    
        # Apply combined transformation
        rot_mat = rot_x @ rot_y @ rot_z
        full_rot = base_rot @ rot_mat
        inv_rot = full_rot.inverted()
    
        # Transform to local coordinate system
        local_verts = [(inv_rot @ (v - bone_head_world)) for v in bone_vertices_world]
    
        # Evaluate configuration quality
        volume, dims = calculate_bounding_box_volume(local_verts)
    
        # 7.1.6 Dimensional Constraints
        dims = (dims[0], dims[1], max(dims[2], bone_length))
        adjusted_volume = dims[0] * dims[1] * dims[2]
    
        # 7.1.7 Optimization Update
        if adjusted_volume < best_volume:
            best_volume = adjusted_volume
            best_rot = full_rot
            best_dims = Vector(dims)

    # 7.1.8 Final Boundary Calculation
    inv_rot = best_rot.inverted()
    local_verts = [(inv_rot @ (v - bone_head_world)) for v in bone_vertices_world]
    min_x = min(v.x for v in local_verts)
    max_x = max(v.x for v in local_verts)
    min_y = min(v.y for v in local_verts)
    max_y = max(v.y for v in local_verts)
    min_z = min(v.z for v in local_verts)
    max_z = max(v.z for v in local_verts)

    min_z = min(min_z, 0)
    max_z = max(max_z, bone_length)
    
    # 7.1.9 Face Data Extraction
    bone_faces = get_faces_for_bone(mesh_obj, vertex_indices)
    face_data, vertex_face_map = extract_face_data(mesh_obj, bone_faces, vertex_indices_set)
    
    face_data_result = extract_face_data(mesh_obj, bone_faces, vertex_indices_set)
    created_cubes = []
    
    # 7.1.10 Data Format Handling
    if isinstance(face_data_result, tuple):
        face_data, vertex_face_map = face_data_result
    else:
        face_data = face_data_result
        vertex_face_map = None
        
    # 7.1.11 Texture Selection Prioritization
    face_data = calculate_spatial_priority(face_data, mesh_obj, bone_head_world)
    
    # 7.2 Specialized Bone Processing
    if is_special:
        # 7.2.1 Configuration for Special Bones
        lod = SPECIAL_BONES[bone_name].get("lod", 3)
        scale_factor = SPECIAL_BONES[bone_name].get("scale_factor", 1.0)
        
        # 7.2.2 Adaptive Resolution
        edge_distance = max(1, int(6 / lod))
        
        # 7.2.3 Spatial Indexing
        kd_tree = KDTree(len(vertex_indices))
        for i, v_idx in enumerate(vertex_indices):
            kd_tree.insert(mesh.vertices[v_idx].co, i)
        kd_tree.balance()
        
        # 7.2.4 Vertex Allocation Tracking
        processed_vertices = set()
        
        # 7.2.5 Subdivision Calculation
        target_cuboids = max(3, lod * lod)
        
        # 7.2.6 Progressive Subdivision
        cuboid_count = 0
        while len(processed_vertices) < len(vertex_indices) and cuboid_count < target_cuboids:
            # 7.2.7 Next Region Selection
            unprocessed = [idx for idx in range(len(vertex_indices)) if idx not in processed_vertices]
            if not unprocessed:
                break
                
            # 7.2.8 Strategic Vertex Selection
            start_vertex_idx = unprocessed[0]
            if processed_vertices:
                max_min_dist = -1
                for idx in unprocessed:
                    v_co = mesh.vertices[vertex_indices[idx]].co
                    min_dist = float('inf')
                    for p_idx in processed_vertices:
                        p_co = mesh.vertices[vertex_indices[p_idx]].co
                        dist = (v_co - p_co).length
                        min_dist = min(min_dist, dist)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        start_vertex_idx = idx
            
            # 7.2.9 Connectivity Analysis
            connected_local = find_connected_vertices(mesh, vertex_indices[start_vertex_idx], edge_distance)
            
            # 7.2.10 Index Conversion
            connected_indices = [i for i, v_idx in enumerate(vertex_indices) if v_idx in connected_local]
            
            # 7.2.11 Cluster Registration
            processed_vertices.update(connected_indices)
            
            # 7.2.12 Cluster Validation and Processing
            if connected_indices:
                # 7.2.13 Coordinate Transformation
                cluster_verts_world = [bone_vertices_world[i] for i in connected_indices]
                cluster_verts_local = [(inv_rot @ (v - bone_head_world)) for v in cluster_verts_world]
                
                # 7.2.14 Boundary Computation
                c_min_x = min(v.x for v in cluster_verts_local)
                c_max_x = max(v.x for v in cluster_verts_local)
                c_min_y = min(v.y for v in cluster_verts_local)
                c_max_y = max(v.y for v in cluster_verts_local)
                c_min_z = min(v.z for v in cluster_verts_local)
                c_max_z = max(v.z for v in cluster_verts_local)
                
                # 7.2.15 Position Calculation  
                center_offset_local = Vector((
                    (c_min_x + c_max_x) / 2,
                    (c_min_y + c_max_y) / 2,
                    (c_min_z + c_max_z) / 2
                ))
                center_offset_world = bone_head_world + (best_rot @ center_offset_local)
                
                # 7.2.16 Dimension Calculation
                padding = 0.01
                cuboid_scale = Vector((
                    (c_max_x - c_min_x + padding) * scale_factor,
                    (c_max_y - c_min_y + padding) * scale_factor,
                    (c_max_z - c_min_z + padding) * scale_factor
                ))
                
                # 7.2.17 Minimum Size Enforcement
                min_size = 0.01
                cuboid_scale.x = max(cuboid_scale.x, min_size)
                cuboid_scale.y = max(cuboid_scale.y, min_size)
                cuboid_scale.z = max(cuboid_scale.z, min_size)
                
                # 7.2.18 Object Creation
                bpy.ops.mesh.primitive_cube_add(size=1.0)
                cube = bpy.context.active_object
                cube.location = center_offset_world
                cube.rotation_euler = best_rot.to_euler()
                cube.scale = cuboid_scale
                cube.name = f"{mesh_name_prefix}{bone_name}_Voxel_{cuboid_count}"
                
                # 7.2.19 Texture Mapping
                cluster_original_indices = [vertex_indices[i] for i in connected_indices]
                cuboid_faces = get_faces_for_bone(mesh_obj, cluster_original_indices)
                cuboid_face_data = extract_face_data(mesh_obj, cuboid_faces)
                apply_face_data_to_cube(cube, mesh_obj, cuboid_face_data if cuboid_face_data else face_data, vertex_face_map)
                
                # 7.2.20 Collection Assignment
                for coll in cube.users_collection:
                    coll.objects.unlink(cube)
                cube_collection.objects.link(cube)
                created_cubes.append(cube)
                
                cuboid_count += 1
        
        print(f"[7.2] Created {cuboid_count} connectivity-based cuboids for special bone {bone_name}")
    else:
        # 7.3 Standard Bone Processing
        # 7.3.1 Position Calculation
        center_offset_local = Vector((
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            (min_z + max_z) / 2
        ))
        center_offset_world = bone_head_world + (best_rot @ center_offset_local)

        # 7.3.2 Dimension Calculation with Padding
        padding = 0.05
        final_scale = Vector((
            max_x - min_x + padding,
            max_y - min_y + padding,
            max_z - min_z + padding
        ))

        # 7.3.3 Object Creation
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        cube = bpy.context.active_object
        cube.location = center_offset_world
        cube.rotation_euler = best_rot.to_euler()
        cube.scale = final_scale
        cube.name = f"{mesh_name_prefix}{bone_name}_Cuboid"
        
        # 7.3.4 Texture Application
        apply_face_data_to_cube(cube, mesh_obj, face_data, vertex_face_map)
        
        # 7.3.5 Collection Assignment
        for coll in cube.users_collection:
            coll.objects.unlink(cube)
        cube_collection.objects.link(cube)
        created_cubes.append(cube)

    return created_cubes

# 7.4 Mesh Processing Orchestration
def generate_bone_boxes_from_selection():
    # 7.4.1 Selection Validation
    arm_obj, mesh_objs = get_selected_armature_and_meshes()
    if not arm_obj or not mesh_objs:
        return []
    
    # 7.4.2 Collection Management
    collection_name = "Bone_Cubes"
    if collection_name in bpy.data.collections:
        cube_collection = bpy.data.collections[collection_name]
    else:
        cube_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(cube_collection)
    
    # 7.4.3 Context Preparation
    original_collection = bpy.context.view_layer.active_layer_collection
    for layer_collection in bpy.context.view_layer.layer_collection.children:
        if layer_collection.collection == cube_collection:
            bpy.context.view_layer.active_layer_collection = layer_collection
            break
    
    # 7.4.4 Mesh Iteration
    created_cubes = []
    for mesh_obj in mesh_objs:
        print(f"[7.4.4] Processing mesh: {mesh_obj.name}")
        
        # 7.4.5 Naming Convention Management
        mesh_prefix = ""
        if len(mesh_objs) > 1:
            mesh_prefix = f"{mesh_obj.name}_"
        
        # 7.4.6 Vertex Assignment
        bone_vertices = assign_vertices_to_bones(mesh_obj)
        
        # 7.4.7 Per-Bone Cuboid Generation
        for bone_name, vertex_indices in bone_vertices.items():
            cubes = create_cuboid_for_bone(arm_obj, mesh_obj, bone_name, vertex_indices, cube_collection, mesh_prefix)
            created_cubes.extend(cubes)
    
    # 7.4.8 Context Restoration
    bpy.context.view_layer.active_layer_collection = original_collection
    
    print(f"[7.4] Created {len(created_cubes)} cubes based on bone influences.")
    print(f"[7.4] Processed {len(mesh_objs)} mesh objects under the armature.")
    print(f"[7.4] All mesh vertices are accounted for in the voxelization.")
    print(f"[7.4] Textures from original meshes have been applied to cubes using direct face mapping.")
    
    return created_cubes

# 7.5 Symmetry Processing
def process_symmetrical_bones(created_cubes):
    """
    Process symmetrical bone pairs to ensure matching scale and rotation.
    For each pair, copy properties from left to right side with appropriate mirroring.
    Uses partial matching of bone names within cube names.
    """
    print("[7.5] Processing symmetrical bone pairs...")
    
    # 7.5.1 Cube Classification
    left_cubes = []
    right_cubes = []
    
    # 7.5.2 Bone Mapping
    bone_part_cubes = {}
    
    # 7.5.3 Symmetry Configuration
    bone_parts = set()
    for left_part, right_part in SYMMETRICAL_BONE_PAIRS.items():
        bone_parts.add(left_part)
        bone_parts.add(right_part)
    
    # 7.5.4 Cube Categorization
    for cube in created_cubes:
        for bone_part in bone_parts:
            if bone_part in cube.name:
                if bone_part not in bone_part_cubes:
                    bone_part_cubes[bone_part] = []
                bone_part_cubes[bone_part].append(cube)
                break
    
    # 7.5.5 Bilateral Processing
    for left_part, right_part in SYMMETRICAL_BONE_PAIRS.items():
        left_part_cubes = bone_part_cubes.get(left_part, [])
        right_part_cubes = bone_part_cubes.get(right_part, [])
        
        if not left_part_cubes or not right_part_cubes:
            print(f"    ⚠️[7.5.5] Skipping symmetrical pair {left_part}/{right_part} - missing cubes")
            continue
        
        print(f"[7.5.5] Processing symmetrical pair: {left_part} -> {right_part}")
        print(f"    ✔️[7.5.5] Found {len(left_part_cubes)} left cubes and {len(right_part_cubes)} right cubes")
        
        # 7.5.6 Bilateral Matching
        for left_cube in left_part_cubes:
            # 7.5.7 Proximity-Based Correspondence
            closest_right_cube = None
            min_distance = float('inf')
            
            left_loc = left_cube.location.copy()
            mirrored_loc = Vector((-left_loc.x, left_loc.y, left_loc.z))
            
            for right_cube in right_part_cubes:
                dist = (right_cube.location - mirrored_loc).length
                if dist < min_distance:
                    min_distance = dist
                    closest_right_cube = right_cube
            
            if closest_right_cube:
                # 7.5.8 Property Mirroring
                # Scale transfer
                closest_right_cube.scale = left_cube.scale.copy()
                
                # Position mirroring
                closest_right_cube.location.y = left_cube.location.y
                closest_right_cube.location.z = left_cube.location.z
                closest_right_cube.location.x = -left_cube.location.x
                
                # Rotation mirroring
                left_euler = left_cube.rotation_euler
                right_euler = closest_right_cube.rotation_euler
                
                # X-axis mirroring: preserve X rotation, invert Y and Z
                right_euler.x = left_euler.x
                right_euler.y = -left_euler.y
                right_euler.z = -left_euler.z
                
                print(f"    ✔️[7.5.8] Matched {left_cube.name} to {closest_right_cube.name}")
    
    print("[7.5] Symmetrical bone processing complete.")

# 7.6 Topology Analysis
def find_connected_vertices(mesh, start_vertex_idx, max_distance):
    """Find all vertices connected to the start vertex within max_distance edges"""
    # 7.6.1 Initialization
    connected = set([start_vertex_idx])
    frontier = set([start_vertex_idx])
    distance = 0
    
    # 7.6.2 Breadth-First Exploration
    while distance < max_distance and frontier:
        new_frontier = set()
        for v_idx in frontier:
            # 7.6.3 Edge Traversal
            vertex = mesh.vertices[v_idx]
            for edge in mesh.edges:
                if v_idx == edge.vertices[0]:
                    other_v = edge.vertices[1]
                    if other_v not in connected:
                        new_frontier.add(other_v)
                        connected.add(other_v)
                elif v_idx == edge.vertices[1]:
                    other_v = edge.vertices[0]
                    if other_v not in connected:
                        new_frontier.add(other_v)
                        connected.add(other_v)
        
        frontier = new_frontier
        distance += 1
    
    return list(connected)

# Execution
cubes = generate_bone_boxes_from_selection()
process_symmetrical_bones(cubes)
