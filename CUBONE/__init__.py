bl_info = {
    "name": "CUBONE",
    "author": "Team Yume:PianoPsychopath",
    "version": (1, 1, 4),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > CUBONE",
    "description": "Generates optimized cuboids based on armature bone influence",
    "warning": "",
    "doc_url": "",
    "category": "Object",
}

import bpy
from bpy.props import (StringProperty, 
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       EnumProperty,
                       CollectionProperty)
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       UIList)

# Import the main script functionality
try:
    from . import cuboid
except ImportError:
    # For development/testing, create placeholder
    class PlaceholderCuboid:
        def __init__(self):
            self.SPECIAL_BONES = {}
            self.WIDE_ROTATION_BONES = []
            self.SYMMETRICAL_BONE_PAIRS = {}
        
        def generate_bone_boxes_from_selection(self):
            return []
            
        def process_symmetrical_bones(self, cubes):
            pass
    
    cuboid = PlaceholderCuboid()

# ------------------------------------------------------
# Helper functions for bone selection dropdowns
# ------------------------------------------------------

def get_armature_bones(self, context):
    """Get all bones from the active armature for dropdown selection"""
    items = []
    
    # Get the active armature if any
    armature = None
    if context.active_object and context.active_object.type == 'ARMATURE':
        armature = context.active_object
    else:
        # Look for any armature in the scene
        for obj in context.scene.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break
    
    # Add bones from the armature
    if armature:
        for bone in armature.data.bones:
            items.append((bone.name, bone.name, f"Bone: {bone.name}"))
    
    # Ensure we have at least one item to avoid Blender errors
    if not items:
        items.append(("None", "No Bones Found", "No armature bones found in the scene"))
        
    return items

# Function for bone pattern shorthand dropdowns
def get_bone_patterns(self, context):
    """Get unique bone name patterns by removing common suffixes like _L/_R"""
    items = []
    patterns = set()
    
    # Get the active armature if any
    armature = None
    if context.active_object and context.active_object.type == 'ARMATURE':
        armature = context.active_object
    else:
        # Look for any armature in the scene
        for obj in context.scene.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break
    
    # Process bone names to get unique patterns
    if armature:
        for bone in armature.data.bones:
            name = bone.name
            # Extract pattern by removing common suffixes
            # This is a simple example - can be enhanced for specific naming conventions
            pattern = name
            if name.endswith("_L") or name.endswith(".L"):
                pattern = name[:-2]
            elif name.endswith("_R") or name.endswith(".R"):
                pattern = name[:-2]
            
            # Add the pattern and the original name
            patterns.add((pattern, name))
    
    # Add full bone names and patterns to items
    # First add full bone names for precise selection
    for bone_name in sorted([b.name for b in armature.data.bones if armature]):
        items.append((bone_name, bone_name, f"Exact bone: {bone_name}"))
    
    # Then add patterns for shorthand selection
    for pattern, example in sorted(patterns):
        if pattern and pattern != example:  # Avoid duplicates
            items.append((pattern, f"{pattern}*", f"Pattern: {pattern}* (e.g., {example})"))
    
    # Ensure we have at least one item to avoid Blender errors
    if not items:
        items.append(("None", "No Bones Found", "No armature bones found in the scene"))
        
    return items

# Function to find matching bones based on pattern
def find_matching_bones(pattern, bone_list):
    """Find all bones that match the given pattern"""
    if pattern == "None":
        return []
        
    matching = []
    for bone in bone_list:
        # Exact match
        if bone == pattern:
            matching.append(bone)
        # Pattern match (bone starts with pattern)
        elif pattern.endswith('*') and bone.startswith(pattern[:-1]):
            matching.append(bone)
        # Left-Right pattern match
        elif (bone.endswith("_L") or bone.endswith(".L")) and bone[:-2] == pattern:
            matching.append(bone)
        elif (bone.endswith("_R") or bone.endswith(".R")) and bone[:-2] == pattern:
            matching.append(bone)
            
    return matching

# ------------------------------------------------------
# Property Groups for UI lists
# ------------------------------------------------------

class CUBONE_SpecialBoneProperties(PropertyGroup):
    name: EnumProperty(
        name="Bone Name",
        description="Name of the bone to apply special settings",
        items=get_armature_bones)
    
    lod: IntProperty(
        name="Level of Detail",
        description="Subdivision detail level for this bone",
        default=8,
        min=1,
        max=32)
    
    scale_factor: FloatProperty(
        name="Scale Factor",
        description="Inflation factor for the bone's voxels",
        default=1.0,
        min=0.1,
        max=5.0,
        precision=2)

class CUBONE_SymmetricalBonePairProperties(PropertyGroup):
    # Use string properties for pattern input
    left_pattern: StringProperty(
        name="Left Pattern",
        description="Pattern for left bones (e.g., 'Arm_1' or 'Arm*')",
        default="")
    
    right_pattern: StringProperty(
        name="Right Pattern",
        description="Pattern for right bones (e.g., 'Arm_1' or 'Arm*')",
        default="")
    
    # Use EnumProperty for dropdown selection
    left_bone: EnumProperty(
        name="Left Bone",
        description="Name of the left side bone",
        items=get_bone_patterns)
    
    right_bone: EnumProperty(
        name="Right Bone",
        description="Name of the right side bone",
        items=get_bone_patterns)
    
    # Toggle for pattern mode vs exact bone mode
    use_pattern: BoolProperty(
        name="Use Pattern",
        description="Use pattern matching for bones instead of exact names",
        default=True)

class CUBONE_WideRotationBoneProperties(PropertyGroup):
    name: EnumProperty(
        name="Bone Name",
        description="Name of the bone requiring wide rotation range",
        items=get_armature_bones)

# ------------------------------------------------------
# Operators for list management
# ------------------------------------------------------

class CUBONE_OT_AddSpecialBone(Operator):
    bl_idname = "cubone.add_special_bone"
    bl_label = "Add Special Bone"
    bl_description = "Add a bone with special processing settings"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        item = context.scene.cubone_special_bones.add()
        # Default values will be set by EnumProperty's first item
        item.lod = 8
        item.scale_factor = 1.0
        context.scene.cubone_special_bones_index = len(context.scene.cubone_special_bones) - 1
        return {'FINISHED'}

class CUBONE_OT_RemoveSpecialBone(Operator):
    bl_idname = "cubone.remove_special_bone"
    bl_label = "Remove Special Bone"
    bl_description = "Remove the selected special bone settings"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        idx = context.scene.cubone_special_bones_index
        if idx >= 0 and len(context.scene.cubone_special_bones) > 0:
            context.scene.cubone_special_bones.remove(idx)
            context.scene.cubone_special_bones_index = min(max(0, idx - 1), 
                                                          len(context.scene.cubone_special_bones) - 1)
        return {'FINISHED'}

class CUBONE_OT_AddWideRotationBone(Operator):
    bl_idname = "cubone.add_wide_rotation_bone"
    bl_label = "Add Wide Rotation Bone"
    bl_description = "Add a bone that requires extended rotation search"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        item = context.scene.cubone_wide_rotation_bones.add()
        # Default values will be set by EnumProperty
        context.scene.cubone_wide_rotation_bones_index = len(context.scene.cubone_wide_rotation_bones) - 1
        return {'FINISHED'}

class CUBONE_OT_RemoveWideRotationBone(Operator):
    bl_idname = "cubone.remove_wide_rotation_bone"
    bl_label = "Remove Wide Rotation Bone"
    bl_description = "Remove the selected wide rotation bone"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        idx = context.scene.cubone_wide_rotation_bones_index
        if idx >= 0 and len(context.scene.cubone_wide_rotation_bones) > 0:
            context.scene.cubone_wide_rotation_bones.remove(idx)
            context.scene.cubone_wide_rotation_bones_index = min(max(0, idx - 1), 
                                                                len(context.scene.cubone_wide_rotation_bones) - 1)
        return {'FINISHED'}

class CUBONE_OT_AddSymmetricalBonePair(Operator):
    bl_idname = "cubone.add_symmetrical_bone_pair"
    bl_label = "Add Symmetrical Bone Pair"
    bl_description = "Add a pair of symmetrical bones"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        item = context.scene.cubone_symmetrical_bone_pairs.add()
        item.use_pattern = True
        context.scene.cubone_symmetrical_bone_pairs_index = len(context.scene.cubone_symmetrical_bone_pairs) - 1
        return {'FINISHED'}

class CUBONE_OT_RemoveSymmetricalBonePair(Operator):
    bl_idname = "cubone.remove_symmetrical_bone_pair"
    bl_label = "Remove Symmetrical Bone Pair"
    bl_description = "Remove the selected symmetrical bone pair"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        idx = context.scene.cubone_symmetrical_bone_pairs_index
        if idx >= 0 and len(context.scene.cubone_symmetrical_bone_pairs) > 0:
            context.scene.cubone_symmetrical_bone_pairs.remove(idx)
            context.scene.cubone_symmetrical_bone_pairs_index = min(max(0, idx - 1), 
                                                                   len(context.scene.cubone_symmetrical_bone_pairs) - 1)
        return {'FINISHED'}

# ------------------------------------------------------
# Main operator to run the cuboid generation
# ------------------------------------------------------

class CUBONE_OT_GenerateCuboids(Operator):
    bl_idname = "cubone.generate_cuboids"
    bl_label = "Generate Bone Cuboids"
    bl_description = "Generate cuboids based on bone influence"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Get all bone names from the active armature
        bone_names = []
        if context.active_object and context.active_object.type == 'ARMATURE':
            bone_names = [bone.name for bone in context.active_object.data.bones]
        else:
            # Look for any armature in the scene
            for obj in context.scene.objects:
                if obj.type == 'ARMATURE':
                    bone_names = [bone.name for bone in obj.data.bones]
                    break
        
        if not bone_names:
            self.report({'ERROR'}, "No armature bones found")
            return {'CANCELLED'}
            
        # Update the global configuration dictionaries in the cuboid module
        
        # 1. Update the special bones dictionary
        special_bones = {}
        for item in context.scene.cubone_special_bones:
            special_bones[item.name] = {"lod": item.lod, "scale_factor": item.scale_factor}
        cuboid.SPECIAL_BONES = special_bones
        
        # 2. Update the wide rotation bones list
        wide_rotation_bones = [item.name.lower() for item in context.scene.cubone_wide_rotation_bones]
        cuboid.WIDE_ROTATION_BONES = wide_rotation_bones
        
        # 3. Update the symmetrical bone pairs dictionary
        symmetrical_bone_pairs = {}
        for item in context.scene.cubone_symmetrical_bone_pairs:
            if item.use_pattern:
                # Get the pattern from input fields
                left_pattern = item.left_bone if item.left_bone else item.left_pattern
                right_pattern = item.right_bone if item.right_bone else item.right_pattern
                
                # Find matching bones
                left_bones = find_matching_bones(left_pattern, bone_names)
                right_bones = find_matching_bones(right_pattern, bone_names)
                
                # Create pairs based on matching patterns
                # This is a simple example - can be enhanced for more complex matching
                if left_pattern == right_pattern:
                    # Same pattern - match L with R bones
                    left_suffixed = [b for b in left_bones if b.endswith("_L") or b.endswith(".L")]
                    right_suffixed = [b for b in right_bones if b.endswith("_R") or b.endswith(".R")]
                    
                    # Match by removing suffix and finding pairs
                    for left in left_suffixed:
                        left_base = left[:-2]  # Remove _L or .L
                        for right in right_suffixed:
                            right_base = right[:-2]  # Remove _R or .R
                            if left_base == right_base:
                                symmetrical_bone_pairs[left] = right
                else:
                    # Different patterns - match all left with all right
                    for left in left_bones:
                        for right in right_bones:
                            symmetrical_bone_pairs[left] = right
            else:
                # Exact bone names
                symmetrical_bone_pairs[item.left_bone] = item.right_bone
                
        cuboid.SYMMETRICAL_BONE_PAIRS = symmetrical_bone_pairs
        
        # Run the main cuboid generation function
        try:
            cubes = cuboid.generate_bone_boxes_from_selection()
            
            # Process symmetrical bones if requested
            if context.scene.cubone_process_symmetry:
                cuboid.process_symmetrical_bones(cubes)
                
            self.report({'INFO'}, f"Successfully generated {len(cubes)} cuboids")
        except Exception as e:
            self.report({'ERROR'}, f"Error generating cuboids: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

# ------------------------------------------------------
# UI Lists
# ------------------------------------------------------

class CUBONE_UL_SpecialBonesList(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon='BONE_DATA')
            row.prop(item, "lod", text="LOD")
            row.prop(item, "scale_factor", text="Scale")
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.prop(item, "name", text="", emboss=False, icon='BONE_DATA')

class CUBONE_UL_WideRotationBonesList(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon='BONE_DATA')
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.prop(item, "name", text="", emboss=False, icon='BONE_DATA')

class CUBONE_UL_SymmetricalBonePairsList(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            
            # Toggle between pattern mode and exact bone mode
            row.prop(item, "use_pattern", text="", toggle=True, icon='VIEWZOOM' if item.use_pattern else 'BONE_DATA')
            
            if item.use_pattern:
                # Pattern mode - show pattern inputs
                row.prop(item, "left_bone", text="", emboss=False, icon='BONE_DATA')
                row.label(text="→")
                row.prop(item, "right_bone", text="", emboss=False, icon='BONE_DATA')
            else:
                # Exact bone mode - show bone dropdowns
                row.prop(item, "left_bone", text="", emboss=False, icon='BONE_DATA')
                row.label(text="→")
                row.prop(item, "right_bone", text="", emboss=False, icon='BONE_DATA')
                
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            if item.use_pattern:
                layout.label(text=f"{item.left_bone}* → {item.right_bone}*")
            else:
                layout.label(text=f"{item.left_bone} → {item.right_bone}")

# ------------------------------------------------------
# Panel UI
# ------------------------------------------------------

class CUBONE_PT_Panel(Panel):
    bl_label = "CUBONE"
    bl_idname = "CUBONE_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CUBONE'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Main operator button
        row = layout.row()
        row.scale_y = 2.0
        row.operator("cubone.generate_cuboids", icon='MOD_WIREFRAME')
        
        layout.separator()
        
        # Special bones section
        box = layout.box()
        box.label(text="Special Bones", icon='MODIFIER')
        
        row = box.row()
        row.template_list("CUBONE_UL_SpecialBonesList", "special_bones_list", scene, 
                          "cubone_special_bones", scene, "cubone_special_bones_index")
        
        col = row.column(align=True)
        col.operator("cubone.add_special_bone", icon='ADD', text="")
        col.operator("cubone.remove_special_bone", icon='REMOVE', text="")
        
        layout.separator()
        
        # Wide rotation bones section
        box = layout.box()
        box.label(text="Wide Rotation Bones", icon='DRIVER')
        
        row = box.row()
        row.template_list("CUBONE_UL_WideRotationBonesList", "wide_rotation_bones_list", scene, 
                          "cubone_wide_rotation_bones", scene, "cubone_wide_rotation_bones_index")
        
        col = row.column(align=True)
        col.operator("cubone.add_wide_rotation_bone", icon='ADD', text="")
        col.operator("cubone.remove_wide_rotation_bone", icon='REMOVE', text="")
        
        layout.separator()
        
        # Symmetrical bone pairs section
        box = layout.box()
        box.label(text="Symmetrical Bone Pairs", icon='MOD_MIRROR')
        
        row = box.row()
        row.template_list("CUBONE_UL_SymmetricalBonePairsList", "symmetrical_bone_pairs_list", scene, 
                          "cubone_symmetrical_bone_pairs", scene, "cubone_symmetrical_bone_pairs_index")
        
        col = row.column(align=True)
        col.operator("cubone.add_symmetrical_bone_pair", icon='ADD', text="")
        col.operator("cubone.remove_symmetrical_bone_pair", icon='REMOVE', text="")
        
        # Add active item's detailed settings
        if len(scene.cubone_symmetrical_bone_pairs) > 0 and scene.cubone_symmetrical_bone_pairs_index >= 0:
            item = scene.cubone_symmetrical_bone_pairs[scene.cubone_symmetrical_bone_pairs_index]
            
            sub_box = box.box()
            sub_box.label(text="Pattern Settings" if item.use_pattern else "Exact Bone Settings")
            
            row = sub_box.row()
            if item.use_pattern:
                row.prop(item, "left_pattern", text="Left Pattern")
                row.prop(item, "right_pattern", text="Right Pattern")
                row = sub_box.row()
                row.label(text="Example: For bones like 'Arm_1_L' and 'Arm_1_R'")
                row = sub_box.row()
                row.label(text="Use pattern 'Arm_1' for both sides")
        
        row = box.row()
        row.prop(scene, "cubone_process_symmetry")

        # Add help text at the bottom
        layout.separator()
        box = layout.box()
        box.label(text="Instructions:", icon='INFO')
        col = box.column(align=True)
        col.label(text="1. Select an armature in object mode")
        col.label(text="2. Add bones to the lists above")
        col.label(text="3. For symmetrical pairs, use patterns like 'Arm_1'")
        col.label(text="4. Click 'Generate Bone Cuboids'")

# ------------------------------------------------------
# Registration
# ------------------------------------------------------

classes = (
    CUBONE_SpecialBoneProperties,
    CUBONE_WideRotationBoneProperties,
    CUBONE_SymmetricalBonePairProperties,
    CUBONE_OT_AddSpecialBone,
    CUBONE_OT_RemoveSpecialBone,
    CUBONE_OT_AddWideRotationBone,
    CUBONE_OT_RemoveWideRotationBone,
    CUBONE_OT_AddSymmetricalBonePair,
    CUBONE_OT_RemoveSymmetricalBonePair,
    CUBONE_OT_GenerateCuboids,
    CUBONE_UL_SpecialBonesList,
    CUBONE_UL_WideRotationBonesList,
    CUBONE_UL_SymmetricalBonePairsList,
    CUBONE_PT_Panel,
)

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    
    # Register properties
    bpy.types.Scene.cubone_special_bones = CollectionProperty(type=CUBONE_SpecialBoneProperties)
    bpy.types.Scene.cubone_special_bones_index = IntProperty(name="Index for special bones list", default=0)
    
    bpy.types.Scene.cubone_wide_rotation_bones = CollectionProperty(type=CUBONE_WideRotationBoneProperties)
    bpy.types.Scene.cubone_wide_rotation_bones_index = IntProperty(name="Index for wide rotation bones list", default=0)
    
    bpy.types.Scene.cubone_symmetrical_bone_pairs = CollectionProperty(type=CUBONE_SymmetricalBonePairProperties)
    bpy.types.Scene.cubone_symmetrical_bone_pairs_index = IntProperty(name="Index for symmetrical bone pairs list", default=0)
    
    bpy.types.Scene.cubone_process_symmetry = BoolProperty(
        name="Process Symmetry",
        description="Apply symmetrical processing to bone pairs",
        default=True
    )

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    
    # Unregister properties
    del bpy.types.Scene.cubone_special_bones
    del bpy.types.Scene.cubone_special_bones_index
    
    del bpy.types.Scene.cubone_wide_rotation_bones
    del bpy.types.Scene.cubone_wide_rotation_bones_index
    
    del bpy.types.Scene.cubone_symmetrical_bone_pairs
    del bpy.types.Scene.cubone_symmetrical_bone_pairs_index
    
    del bpy.types.Scene.cubone_process_symmetry

if __name__ == "__main__":
    register()