from platform import node, python_version_tuple
from sys import path

from numpy.lib.function_base import angle
import bpy
import numpy as np
import math
from typing import Tuple
from mathutils import Matrix, Vector
import glob
import os
import shutil
import argparse
import random

random.seed(125)

import sys
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"



FILTER = False
MULTI_GPU = False
NGPU = 1
CURRENT_GPU =  int(argv[0])
CURRENT_INDEX =  int(argv[1])
print(f'--------------------------------------------- Current GPU : {CURRENT_GPU} --------------------------------------')

os.environ["CUDA_VISIBLE_DEVICES"] = f'{CURRENT_GPU}'


CAMREA1_L = (7*math.sqrt(2), 0, 4.95831)
CAMERA1_R = (1.109319055, 0, 1.5708) # this is the default blender camera parameter
CAMERA1_R2 = (1.5708, 0, 1.5708)

HUMAN_L = (0, 0, 0.762487)
CAMERA_H_R  = (1.136209343, 0, 1.5707963268)
# ------------------------------------------------------------------------------------------
BOY_SCALE = 0.009
BOY_CAMREA = (1.109319055, 0, 0.814927389)
BOY_CAMERA_R = (1.136209, 0, 1.680752)

BOY_SCALE_2 = 0.007
BOY_CAMREA_2 = (1.10932, 0.097, 0.624927)
BOY_CAMERA_R_2 = (1.136209, 0, 1.680752)
# --------------------------------------------------------------------------------------------

CAM_TEMP_P = (0, 0, 5)
CAM_TEMP_R = (0, 0, 0)


# --------------------------------------------------------------------------------------------
# Myriam Object information
MYRIAM_SCALE = 0.0042
MYRIAM_CAM = (1.05932, 0.147, 0.714927)
MYRIAM_CAM_R = (1.1588982, 0, 1.74532925)

# --------------------------------------------------------------------------------------------
# Sophie Object information: Sophie3.fbx
SOPHIE_SCALE = 0.042
SOPHIE_CAM = (1.05932, 0.147, 0.714927)
SOPHIE_CAM_R = (1.17102168, 0, 1.701038032)

# --------------------------------------------------------------------------------------------
# anslemo Object information: Anselmo/Anselmo9.fbx
ANSELMO_SCALE = 0.0032
ANSELMO_CAM = (1.05932, 0.147, 0.714927)
ANSELMO_CAM_R = (1.188474973, 0, 1.701038032)


# --------------------------------------------------------------------------------------------
# anslemo Object information: Anselmo/Anselmo9.f003176
LARS_SCALE = 0.030303
LARS_CAM = (1.05932, 0.147, 0.714927)
LARS_CAM_R = (1.188474973, 0, 1.711510007)
LARS_LOCATION = (0, 0, 0.151952)

def clean_objects() -> None:
    """
    this function is used to clean all the object in the scene
    """
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)


    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)
        
        


def add_camera(name,  index, position =(7.35889 , -6.92579 , 4.95831), rotation=(1.109319055, 0, 0.814927389)):
    """
    this function used to add a camera to the scene with a given name for the camera object.
    and coordinates for the camera position and rotation within the scene
    """
    
    px, py, pz = position[0], position[1], position[2]
    rx, ry , rz = rotation[0], rotation[1], rotation[2]
    
    bpy.ops.object.camera_add(enter_editmode=False, 
        align='VIEW', location=(px, py, pz), 
        rotation=(rx, ry, rz))
        
    
    selected_object = bpy.context.active_object
    selected_object.name = name


    # to be reviewed and removed later
    # bpy.context.object.data.type = 'PERSP'

    
    selected_object.pass_index = index
    
    return selected_object
     



def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


    

def get_calibration_matrix_K_from_blender(camd):

    # print(camd.type)
    # if camd.type != 'PERSP':
    #     raise ValueError('Non-perspective cameras not supported')

    scene = bpy.context.scene
    f_in_mm = camd.lens
    print(f_in_mm)
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)

    print(f'lens : {f_in_mm}')
    print(f'width : {resolution_x_in_px}')
    print(f'sensor width : {camd.sensor_width}')
    print(f'{camd.angle}')

    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K
        
        
        

def add_plane(scale, name, index, p_location=(0,0,0)):
    """
    this function is used to add a plane to the scene given a scale factor and a name.
    the plane is used as support for the mesh that is to be added later
    """
    bpy.ops.mesh.primitive_plane_add()
    
     # set the name of the plane to a custom name created by us
    selected_object = bpy.context.active_object
    selected_object.name = name
    
    bpy.data.objects[name].select_set(True)
    
    
    selected_object.scale[0] = scale
    selected_object.scale[1] = scale
    selected_object.scale[2] = scale
    
    selected_object.location[0] = p_location[0]
    selected_object.location[1] = p_location[1]
    selected_object.location[2] = p_location[2]
    
    selected_object.pass_index = index



def create_texture_node(node_tree: bpy.types.NodeTree, path: str, is_color_data: bool) -> bpy.types.Node:
    # Instantiate a new texture image node
    texture_node = node_tree.nodes.new(type='ShaderNodeTexImage')

    # Open an image and set it to the node
    texture_node.image = bpy.data.images.load(path)

    # Set other parameters
    texture_node.image.colorspace_settings.is_data = False if is_color_data else True

    # Return the node
    return texture_node




def build_pbr_textured_nodes(node_tree: bpy.types.NodeTree,
                             color_texture_path: str = "",
                             metallic_texture_path: str = "",
                             roughness_texture_path: str = "",
                             normal_texture_path: str = "",
                             displacement_texture_path: str = "",
                             ambient_occlusion_texture_path: str = "",
                             scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                             displacement_scale: float = 1.0) -> None:

    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    coord_node = node_tree.nodes.new(type='ShaderNodeTexCoord')
    mapping_node = node_tree.nodes.new(type='ShaderNodeMapping')
    mapping_node.vector_type = 'TEXTURE'
    if bpy.app.version >= (2, 81, 0):
        mapping_node.inputs["Scale"].default_value = scale
    else:
        mapping_node.scale = scale
    node_tree.links.new(coord_node.outputs['UV'], mapping_node.inputs['Vector'])

    if color_texture_path != "":
        texture_node = create_texture_node(node_tree, color_texture_path, True)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        if ambient_occlusion_texture_path != "":
            ao_texture_node = create_texture_node(node_tree, ambient_occlusion_texture_path, False)
            node_tree.links.new(mapping_node.outputs['Vector'], ao_texture_node.inputs['Vector'])
            mix_node = node_tree.nodes.new(type='ShaderNodeMixRGB')
            mix_node.blend_type = 'MULTIPLY'
            node_tree.links.new(texture_node.outputs['Color'], mix_node.inputs['Color1'])
            node_tree.links.new(ao_texture_node.outputs['Color'], mix_node.inputs['Color2'])
            node_tree.links.new(mix_node.outputs['Color'], principled_node.inputs['Base Color'])
        else:
            node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Base Color'])

    if metallic_texture_path != "":
        texture_node = create_texture_node(node_tree, metallic_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Metallic'])

    if roughness_texture_path != "":
        texture_node = create_texture_node(node_tree, roughness_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Roughness'])

    if normal_texture_path != "":
        texture_node = create_texture_node(node_tree, normal_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        normal_map_node = node_tree.nodes.new(type='ShaderNodeNormalMap')
        node_tree.links.new(texture_node.outputs['Color'], normal_map_node.inputs['Color'])
        node_tree.links.new(normal_map_node.outputs['Normal'], principled_node.inputs['Normal'])

    if displacement_texture_path != "":
        texture_node = create_texture_node(node_tree, displacement_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        displacement_node = node_tree.nodes.new(type='ShaderNodeDisplacement')
        displacement_node.inputs['Scale'].default_value = displacement_scale
        node_tree.links.new(texture_node.outputs['Color'], displacement_node.inputs['Height'])
        node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])



def add_material(name: str = "Material",
                 use_nodes: bool = False,
                 make_node_tree_empty: bool = False) -> bpy.types.Material:
    '''
    https://docs.blender.org/api/current/bpy.types.BlendDataMaterials.html
    https://docs.blender.org/api/current/bpy.types.Material.html
    '''

    # TODO: Check whether the name is already used or not

    material = bpy.data.materials.new(name)
    material.use_nodes = use_nodes

    if use_nodes and make_node_tree_empty:
        clean_nodes(material.node_tree.nodes)

    return material



def add_named_material(name: str, scale=(1.0, 1.0, 1.0), displacement_scale: float = 1.0) -> bpy.types.Material:
    mat = add_material(name, use_nodes=True, make_node_tree_empty=True)
    build_pbr_textured_nodes(mat.node_tree,
                                   color_texture_path=texture_paths[name]["color"],
                                   roughness_texture_path=texture_paths[name]["roughness"],
                                   normal_texture_path=texture_paths[name]["normal"],
                                   metallic_texture_path=texture_paths[name]["metallic"],
                                   displacement_texture_path=texture_paths[name]["displacement"],
                                   ambient_occlusion_texture_path=texture_paths[name]["ambient_occlusion"],
                                   scale=scale,
                                   displacement_scale=displacement_scale)
    return mat





def read_mesh_file(path):
    """
    given the path corresponding to a given mesh files, this function imports 
    the mesh into the system via the blender import routine
    """
    ext = path.split('.')[1]

    if ext == 'ply':
        bpy.ops.import_mesh.ply(filepath = path)
    elif ext == 'stl':
        bpy.ops.import_mesh.stl(filepath = path)
    elif ext == 'fbx':
        bpy.ops.import_scene.fbx(filepath = path)
    elif ext == 'gltf':
        bpy.ops.import_scene.gltf(filepath = path)
    elif ext == 'obj':
        bpy.ops.import_scene.obj(filepath = path)
    elif ext == 'x3d':
        bpy.ops.import_scene.x3d(filepath = path)


    
    
   


def add_mesh(mesh_path, scale, name, index, rescale=False, rotate=False, position=False, pose=(0, 0, 0)):
    """
    this function adds a mesh to the scene given the path,
    a scale and a name are also used
    """
    # this code is only applicable for the cinema 4d objects (fbx)
    read_mesh_file(path=mesh_path)

    # unselect all the object and delete the weird camera object from cimera 4d
    bpy.ops.object.select_all(action='DESELECT')
    keys = bpy.data.objects.keys()

    if 'CINEMA_4D_Editor' in keys:
        bpy.data.objects['CINEMA_4D_Editor'].select_set(True) # Blender 2.8x
        bpy.ops.object.delete() 

    # get the object name from blender internal objects list
    name = bpy.data.objects[0].name

    # selected_object = bpy.context.active_object
    bpy.data.objects[name].select_set(True)
    
    
    # increase the scale of the object
    if rescale:
        bpy.data.objects[name].scale[0] = scale
        bpy.data.objects[name].scale[1] = scale
        bpy.data.objects[name].scale[2] = scale


    if position:
        bpy.data.objects[name].location[0] = pose[0]
        bpy.data.objects[name].location[1] = pose[1]
        bpy.data.objects[name].location[2] = pose[2]
    

    # do a rotation along the x-axis to position the mesh standing
    if rotate:
        bpy.data.objects[name].rotation_euler[0] = 1.5708

    bpy.data.objects[name].pass_index = index
    
    return bpy.data.objects[name], name




def add_human_mesh(mesh_path, scale, name,location, index, rescale=False, rotate=False):
    """
    this function adds a mesh to the scene given the path,
    a scale and a name are also used
    """
    # bpy.ops.import_mesh.ply(filepath = mesh_path)
    read_mesh_file(path=mesh_path)
    selected_object = bpy.context.active_object
    
    
    # bpy.data.objects[name].select_set(True)
    # set the inserted mesh location
    selected_object.location[0] = location[0]
    selected_object.location[1] = location[1]
    selected_object.location[2] = location[2]
    
    
    # increase the scale of the object
    if rescale:
        selected_object.scale[0] = scale
        selected_object.scale[1] = scale
        selected_object.scale[2] = scale
    

    # do a rotation along the x-axis to position the mesh standing
    if rotate:
        selected_object.rotation_euler[0] = 1.5708

    selected_object.pass_index = index

    # create the material and adding a vertex color node connect them

    # change the shader type to object  activate material node usage
    # bpy.context.space_data.shader_type = 'OBJECT'
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    node_tree = material.node_tree


    # get the node tree
    principled_node = node_tree.nodes.get('Principled BSDF"', None)
    vertex_color = node_tree.nodes.get('Vertex Color', None)
    output_node = node_tree.nodes.get('Material Output', None)

    if principled_node is None:
        principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

    if output_node is None:
        output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    if vertex_color is None:
        vertex_color = node_tree.nodes.new(type='ShaderNodeVertexColor')

    # connects all the nodes post creation
    node_tree.links.new(vertex_color.outputs['Color'], principled_node.inputs['Base Color'])
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])


    # add the material to the object
    selected_object.data.materials.append(bpy.data.materials[name])
    
    return selected_object
    
    
def build_environment_texture_background(world: bpy.types.World, hdri_path: str, rotation: float = 0.0) -> None:
    """
    this function builds the environment map which will be used as illumination for the entire scene.
    am image path, together with rotation factor are provided to the function as input
    """
    world.use_nodes = True
    node_tree = world.node_tree
    
    # retrieve the environment node and the mapping node to see of they exist within the scene
    environment_texture_node = node_tree.nodes.get("Environment Texture", None)
    mapping_node = node_tree.nodes.get("Mapping", None)
    
    
    # if the environment or mapping node are not existent in the scene we create them
    if environment_texture_node is None:
        environment_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        
    if mapping_node is None:
        mapping_node = node_tree.nodes.new(type="ShaderNodeMapping")
        
        
    environment_texture_node.image = bpy.data.images.load(hdri_path)
    
    if bpy.app.version >= (2, 81, 0):
        mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, rotation)
    else:
        mapping_node.rotation[2] = rotation

    
    tex_coord_node = node_tree.nodes.get('Texture Coordinate', None)
    
    if tex_coord_node is None:
        tex_coord_node = node_tree.nodes.new(type="ShaderNodeTexCoord")
    
    
    # ---------------------------------------------------------------------
    background_node = node_tree.nodes.get('Background', None)
    world_output = node_tree.nodes.get('World Output', None)
    
    
    if world_output is None:
        world_output = node_tree.nodes.new(type="ShaderNodeOutputWorld")
            

    node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    node_tree.links.new(mapping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
    node_tree.links.new(environment_texture_node.outputs["Color"], node_tree.nodes["World Output"].inputs["Surface"])

    # arrange_nodes(node_tree)
    return mapping_node
    
    
def set_output_properties(scene: bpy.types.Scene,
                      resolution_percentage: int = 100,
                      output_file_path: str = "",
                      res_x: int = 512,
                      res_y: int = 512) -> None:
    """
    this function sets the output propertiesof the scene which will later be used
    during the rendering process. also an output path is included
    """
                          
    scene.render.resolution_percentage = resolution_percentage
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    if output_file_path:
        scene.render.filepath = output_file_path
        
        


def set_cycles_renderer(scene: bpy.types.Scene,
                        camera_object: bpy.types.Object,
                        num_samples: int,
                        use_denoising: bool = True,
                        use_motion_blur: bool = False,
                        use_transparent_bg: bool = False,
                        prefer_cuda_use: bool = True,
                        use_adaptive_sampling: bool = False) -> None:
    scene.camera = camera_object

    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = use_motion_blur

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.use_adaptive_sampling = use_adaptive_sampling
    scene.cycles.samples = num_samples

    # Enable GPU acceleration
    # Source - https://blender.stackexchange.com/a/196702
    if prefer_cuda_use:
        bpy.context.scene.cycles.device = "GPU"

        # Change the preference setting
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

    # Call get_devices() to let Blender detects GPU device (if any)
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # Let Blender use all available devices, include GPU and CPU
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1

    # Display the devices to be used for rendering
    print("----")
    print("The following devices will be used for path tracing:")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        print("- {}".format(d["name"]))
    print("----")




def render_image(output_path, mapping, mesh, scene, angles):

    # render while rotating the object and the illumination or either of the two one after the other
    # angles = [i for i in range(rg[0], rg[1], rg[2])]
    for angle in angles:
        
        # rotate the mesh by a certain degree
        mesh.rotation_euler[2] = math.radians(angle)
        
        for angle0 in angles:
            current_angle = angle0
            
            # perform the rotation on the mapping node for light source rotation
            mapping.inputs[2].default_value[2] = math.radians(current_angle)

            name = f'image_{angle}_{current_angle}.png'
            scene.render.filepath = f'{output_path}/{name}'
                
            bpy.ops.render.render(write_still=True, use_viewport=True) 



def render_others(output_path, mesh, scene, rg):

    # render while rotating the object and the illumination or either of the two one after the other
    angles = [i for i in range(rg[0], rg[1], rg[2])]
    for angle in angles:
        
        # rotate the mesh by a certain degree
        mesh.rotation_euler[2] = math.radians(angle)

        name = f'image_{angle}.png'
        scene.render.filepath = f'{output_path}/{name}'
            
        bpy.ops.render.render(write_still=True, use_viewport=True)  



def dist2depth_img(dist_img, focal=50*1e-5):

    img_width = dist_img.shape[1]
    img_height = dist_img.shape[0]

    # Get x_i and y_i (distances from optical center)
    cx = img_width // 2
    cy = img_height // 2

    xs = np.arange(img_width) - cx
    ys = np.arange(img_height) - cy
    xis, yis = np.meshgrid(xs, ys)

    depth = np.sqrt(
        dist_img ** 2 / (
            (xis ** 2 + yis ** 2) / (focal ** 2) + 1
        )
    )

    return depth



def render_depth(output_path, mesh, scene, rg, cam_data):

    # extract camera data
    f_in_mm = cam_data.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    sensor_width = cam_data.sensor_width

    focal_length_p_in_x = resolution_x_in_px * (f_in_mm / sensor_width)

    # render while rotating the object and the illumination or either of the two one after the other
    angles = [i for i in range(rg[0], rg[1], rg[2])]
    for angle in angles:
        
        # rotate the mesh by a certain degree
        mesh.rotation_euler[2] = math.radians(angle)

        name = f'image_{angle}.png'
        scene.render.filepath = f'{output_path}/{name}'
            
        bpy.ops.render.render(write_still=True, use_viewport=True)  

        # save original numpy file
        depth = np.asarray(bpy.data.images["Viewer Node"].pixels)
        depth = np.reshape(depth, (480, 640, 4))
        depth = depth[:, :, 0]
        depth = np.flipud(depth)

        # converting the distance depth to a real depth map
        depth = dist2depth_img(depth, focal=focal_length_p_in_x)

        name2 = f'image_{angle}.npy'
        np.save(f'{output_path}/{name2}', depth)





def render_background(output_path, mapping, scene):

    # render while rotating the object and the illumination or either of the two one after the other
    angles = [i for i in range(0, 360, 45)]
    for angle in angles:
        
        current_angle = angle
        
        # perform the rotation on the mapping node for light source rotation
        mapping.inputs[2].default_value[2] = math.radians(current_angle)

        name = f'background_{current_angle}.png'
        scene.render.filepath = f'{output_path}/{name}'
            
        bpy.ops.render.render(write_still=True, use_viewport=True) 




def setup_for_alpha_rendering(object_index):
    
    # setup nodes object id for the render layer and make it available
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True


    # check if the corresponding node exist and create them
    # world.use_nodes = True
    if bpy.context.scene.use_nodes is False:
        bpy.context.scene.use_nodes = True

    # node_tree = world.node_tree
    node_tree = bpy.data.scenes['Scene'].node_tree
    
    
    render_layer_node =  node_tree.nodes.get('Render Layers', None)
    id_mask_node = node_tree.nodes.get('ID Mask', None)
    composite_node = node_tree.nodes.get('Composite', None)
    
    
    # if the required compositor nodes are not availabe we create them
    if render_layer_node is None:
        render_layer_node = node_tree.nodes.new(type="CompositorNodeRLayers")
    
    if id_mask_node is None:
        id_mask_node = node_tree.nodes.new(type="CompositorNodeIDMask")
        
    if composite_node is None:
        composite_node = node_tree.nodes.new(type="CompositorNodeComposite")
        
        
    # set the id for the current object to be rendered
    id_mask_node.index = object_index
    
    
    # connect the compositor nodes to each other
    node_tree.links.new(render_layer_node.outputs['IndexOB'], id_mask_node.inputs['ID value'])
    node_tree.links.new(id_mask_node.outputs['Alpha'], composite_node.inputs['Image'])
    
    
   
   
   
def setup_composite_node():
    
    # setup nodes object id for the render layer and make it available
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True


    # check if the corresponding node exist and create them
    # world.use_nodes = True
    
    if bpy.context.scene.use_nodes is False:
        bpy.context.scene.use_nodes = True
        

    # node_tree = world.node_tree
    node_tree = bpy.data.scenes['Scene'].node_tree
    
    
    render_layer_node =  node_tree.nodes.get('Render Layers', None)
    composite_node = node_tree.nodes.get('Composite', None)
    
    
    # if the required compositor nodes are not availabe we create them
    if render_layer_node is None:
        render_layer_node = node_tree.nodes.new(type="CompositorNodeRLayers")
        
    if composite_node is None:
        composite_node = node_tree.nodes.new(type="CompositorNodeComposite")
    
    
    # connect the compositor nodes to each other
    node_tree.links.new(render_layer_node.outputs['Image'], composite_node.inputs['Image'])
    
    
    
    
def setup_depth_rendering():
    
    # setup nodes object id for the render layer and make it available
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True


    # check if the corresponding node exist and create them
    # world.use_nodes = True
    if bpy.context.scene.use_nodes is False:
        bpy.context.scene.use_nodes = True

    # node_tree = world.node_tree
    node_tree = bpy.data.scenes['Scene'].node_tree
    
    
    render_layer_node =  node_tree.nodes.get('Render Layers', None)
    normalize_node = node_tree.nodes.get('Normalize', None)
    composite_node = node_tree.nodes.get('Composite', None)
    
    
    # if the required compositor nodes are not availabe we create them
    if render_layer_node is None:
        render_layer_node = node_tree.nodes.new(type="CompositorNodeRLayers")
    
    if normalize_node is None:
        normalize_node = node_tree.nodes.new(type="CompositorNodeNormalize")
        
    if composite_node is None:
        composite_node = node_tree.nodes.new(type="CompositorNodeComposite")
    
    
    # connect the compositor nodes to each other
    node_tree.links.new(render_layer_node.outputs['Depth'], normalize_node.inputs['Value'])
    node_tree.links.new(normalize_node.outputs['Value'], composite_node.inputs['Image'])





def setup_for_depth():

    # setup nodes object id for the render layer and make it available
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True


    # check if the corresponding node exist and create them
    # world.use_nodes = True
    if bpy.context.scene.use_nodes is False:
        bpy.context.scene.use_nodes = True

    # node_tree = world.node_tree
    node_tree = bpy.data.scenes['Scene'].node_tree
    
    
    render_layer_node =  node_tree.nodes.get('Render Layers', None)
    # normalize_node = node_tree.nodes.get('Normalize', None)
    composite_node = node_tree.nodes.get('Composite', None)

    map_range_node = node_tree.nodes.get('Map Range', None)
    viewer_node = node_tree.nodes.get('Viewer', None)
    
    
    # if the required compositor nodes are not availabe we create them
    if render_layer_node is None:
        render_layer_node = node_tree.nodes.new(type="CompositorNodeRLayers")
    
    # if normalize_node is None:
    #     normalize_node = node_tree.nodes.new(type="CompositorNodeNormalize")
        
    if composite_node is None:
        composite_node = node_tree.nodes.new(type="CompositorNodeComposite")

    if map_range_node is None:
        map_range_node = node_tree.nodes.new(type="CompositorNodeMapRange")

    if viewer_node is None:
        viewer_node = node_tree.nodes.new(type="CompositorNodeViewer")
    
    
    # connect the compositor nodes to each other
    # render later to composite node
    node_tree.links.new(render_layer_node.outputs['Image'], composite_node.inputs['Image'])

    # render layer to map rande node
    map_range_node.inputs[1].default_value = 0
    map_range_node.inputs[2].default_value = 100
    map_range_node.inputs[3].default_value = 0
    map_range_node.inputs[4].default_value = 1
    node_tree.links.new(render_layer_node.outputs['Depth'], map_range_node.inputs['Value'])

    # map range to viewer node
    node_tree.links.new(map_range_node.outputs['Value'], viewer_node.inputs['Image'])

    # node_tree.links.new(render_layer_node.outputs['Depth'], normalize_node.inputs['Value'])
    # node_tree.links.new(normalize_node.outputs['Value'], composite_node.inputs['Image'])

    

def remove_mesh_color(name):

    # get the current object material
    materials = bpy.data.materials
    object_material = materials[name]
    object_material.use_nodes = True

    # get the node tree and remove the mesh color
    node_tree = object_material.node_tree
    vertex_color_node = node_tree.nodes['Vertex Color']
    
    link = vertex_color_node.outputs['Color'].links[0]
    node_tree.links.remove(link)



def remove_base_color(name):

    # get the current object material
    materials = bpy.data.materials
    object_material = materials[name]
    object_material.use_nodes = True

    # get the node tree and remove the mesh color
    node_tree = object_material.node_tree
    principled_bsdf = node_tree.nodes['Principled BSDF']
    
    link = principled_bsdf.inputs['Base Color'].links[0]
    node_tree.links.remove(link)

    # set the default mesh color to 0.588
    principled_bsdf.inputs['Base Color'].default_value[0] = 0.588
    principled_bsdf.inputs['Base Color'].default_value[1] = 0.588
    principled_bsdf.inputs['Base Color'].default_value[2] = 0.588
    



def add_base_color(name):

# get the current object material
    materials = bpy.data.materials
    object_material = materials[name]
    object_material.use_nodes = True

    # get the node tree and remove the mesh color
    node_tree = object_material.node_tree
    principled_node = node_tree.nodes.get('Principled BSDF"', None)
    image_texture = node_tree.nodes.get('Image Texture', None)

    if principled_node is None:
        principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

    if image_texture is None:
        image_texture = node_tree.nodes.new(type='ShaderNodeTexEnvironment')

    # connects all the nodes post creation
    node_tree.links.new(image_texture.outputs['Color'], principled_node.inputs['Base Color'])




def add_mesh_color(name):

# get the current object material
    materials = bpy.data.materials
    object_material = materials[name]
    object_material.use_nodes = True

    # get the node tree and remove the mesh color
    node_tree = object_material.node_tree
    principled_node = node_tree.nodes.get('Principled BSDF"', None)
    vertex_color = node_tree.nodes.get('Vertex Color', None)
    output_node = node_tree.nodes.get('Material Output', None)

    if principled_node is None:
        principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

    if output_node is None:
        output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    if vertex_color is None:
        vertex_color = node_tree.nodes.new(type='ShaderNodeVertexColor')

    # connects all the nodes post creation
    node_tree.links.new(vertex_color.outputs['Color'], principled_node.inputs['Base Color'])
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])




def render_images(scene, output_path_image, camera_object, num_samples, mapping, mesh, angles, name):

    #sets the output properties of the scne to rendered
    # path_image = f'{output_file_path}/image.png'
    set_output_properties(scene, 100, output_path_image, res_x = 640, res_y = 480)
    
    # setup the composite layers before rendering
    setup_composite_node()

    # add_mesh_color('soccerc_hips000001')
    add_base_color('Default')
    
    # configures the cycle rendering engine
    set_cycles_renderer(scene, camera_object, num_samples)
    
    # first render the rgb images
    render_image(output_path_image, mapping, mesh, scene, angles)



def render_foreground_shading(scene, output_path_shading, camera_object, num_samples, mapping, mesh, rg, name):
    #sets the output properties of the scne to rendered
    # path_image = f'{output_file_path}/image.png'
    set_output_properties(scene, 100, output_path_shading, res_x = 640, res_y = 480)
    
    # setup the composite layers before rendering
    setup_composite_node()

    # remove mesh color prior to rendering foreground shading image
    # remove_mesh_color('soccerc_hips000001')
    remove_base_color('Default')
    
    # configures the cycle rendering engine
    set_cycles_renderer(scene, camera_object, num_samples)
    
    # render the shading images
    render_image(output_path_shading, mapping, mesh, scene, rg)



def render_foreground_mask(scene, output_path_mask_foreground, camera_object, num_samples, mesh, rg):

    set_output_properties(scene, 100, output_path_mask_foreground, res_x = 640, res_y = 480)
    
    setup_for_alpha_rendering(1)
    
    # configures the cycle rendering engine
    set_cycles_renderer(scene, camera_object, num_samples)
    
    # otherforms of rendering
    render_others(output_path_mask_foreground, mesh, scene, rg)


def render_background_mask(scene, output_path_mask_background, camera_object, num_samples, mesh, rg):

    set_output_properties(scene, 100, output_path_mask_background, res_x = 640, res_y = 480)
    
    setup_for_alpha_rendering(2)
    
    # configures the cycle rendering engine
    set_cycles_renderer(scene, camera_object, num_samples)
    
    # other form of rendering
    render_others(output_path_mask_background, mesh, scene, rg)
        


def render_depth_map(scene, output_path_depth, camera_object, num_samples, mesh, rg):

    # path_image = f'{output_file_path}/depth.png'
    set_output_properties(scene, 100, output_path_depth, res_x = 640, res_y = 480)
    
    # setup_depth_rendering()
    setup_for_depth()
    
    # configures the cycle rendering engine
    set_cycles_renderer(scene, camera_object, num_samples)
    
    # depth rendering function
    render_depth(output_path_depth, mesh, scene, rg, camera_object.data)


def render_background_image(camera_object, scene, output_path_background, num_samples, mapping):

    camera_object.rotation_euler[0] = math.radians(90)

    set_output_properties(scene, 100, output_path_background, res_x = 640, res_y = 480)
    
    setup_composite_node()
    
    # configures the cycle rendering engine
    set_cycles_renderer(scene, camera_object, num_samples)
    
    # background rendering function
    render_background(output_path_background, mapping, scene)


def create_or_recreate_folders(folder):
    """
    deletes existing folder if they already exist and
    recreates then. Only valid for training mode. does not work in
    resume mode
    :return:
    """
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        os.mkdir(folder)
    else:
        os.mkdir(folder)



def do_hdr_filtering(hdr_images_path, generated_images_path):
    hdr_images_list = os.listdir(hdr_images_path)
    generated_images_list = os.listdir(generated_images_path)

    
    for i in range(len(generated_images_list)):
        current_hdr = generated_images_list[i]
        if f'{current_hdr}.exr' in hdr_images_list:
            hdr_images_list.remove(f'{current_hdr}.exr')

    new_list = []
    for i in range(len(hdr_images_list)):
        current_image = hdr_images_list[i]
        current_path = glob.glob(os.path.join(hdr_images_path, current_image))
        new_list.append(current_path[0])

    
    return new_list


def main():
    clean_objects()
    for m in bpy.data.materials:
        bpy.data.materials.remove(m)
    
    # --------------------------------------------- dataset rendering path --------------------------------------------------
    cam_z = 4.95831
    mesh_path = 'your mesh path'
    hdr_images_path = 'your hdr path'
    # generated_images_path = './anselmo10/images'
    hdr_paths = glob.glob(f'{hdr_images_path}/*.exr')

    # randomly sampling hdr paths
    hdr_indexes = random.sample(range(0, len(hdr_paths) - 1), len(hdr_paths) // 2)
    
    
    

    # if FILTER:
    #     hdr_paths = do_hdr_filtering(hdr_images_path, generated_images_path)
    
    hdr_paths.sort()

    # perform slicing for gpu assignement
    if MULTI_GPU:
        n = NGPU
        splited = [hdr_paths[i::n] for i in range(n)]
        hdr_paths = splited[CURRENT_INDEX]



    # -----------------------------------------------------------------------------------------------------------------------
    # save paths for different types of images
    output_path_image = './images'
    output_path_shading = './shading'




    num_samples = 100
    rg = (0, 360, 45)
    angles_list = [i for i in range(rg[0], rg[1], rg[2])]

    
    
    # --------------------------------------------------------------------------------------------------------------------------


    # add the cameras to the scene
    camera_object = add_camera(name='cam1', position =LARS_CAM, rotation=LARS_CAM_R, index=3)
    
    # add objects and meshes
    plane_location = (0, 0, 0.142924)
    add_plane(20, 'plane', index=2, p_location=plane_location)

    mesh, name = add_mesh(mesh_path,
                        LARS_SCALE,
                        '10462_m_Andy_100k',
                        index=1, rescale=True,
                        rotate=False,
                        position=True,
                        pose=LARS_LOCATION)
    # mesh = add_human_mesh(mesh_path, 0.0013, 'soccerc_hips000001', location=HUMAN_L, index=1, rescale=True, rotate=True)

    # bug code : resising the plane (to be removed later)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['plane'].scale[0] = 20
    bpy.data.objects['plane'].scale[1] = 20
    bpy.data.objects['plane'].scale[2] = 20
     

    # building the scene
    scene = bpy.data.scenes["Scene"]
    world = scene.world
    
    
    for i in range(len(hdr_indexes)):
        current_hdr_index = hdr_indexes[i]

        if len(angles_list) > 1:
            samp_ang_index = random.sample(range(0, len(angles_list) - 1), len(angles_list) // 2)
            angles = [angles_list[ind] for ind in samp_ang_index]

        # angles = angles[0:1]
        

        if i > 0:
            clean_objects()
            # add the cameras to the scene
            camera_object = add_camera(name='cam1', position =LARS_CAM, rotation=LARS_CAM_R, index=3)
            
            # add objects and meshes
            add_plane(20, 'plane', index=2, p_location=plane_location)

            mesh, name = add_mesh(mesh_path,
                        LARS_SCALE,
                        '10462_m_Andy_100k',
                        index=1, rescale=True,
                        rotate=False,
                        position=True,
                        pose=LARS_LOCATION)
            

            # bug code : resising the plane (to be removed later)
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects['plane'].scale[0] = 20
            bpy.data.objects['plane'].scale[1] = 20
            bpy.data.objects['plane'].scale[2] = 20
            
            # building the scene
            scene = bpy.data.scenes["Scene"]
            world = scene.world

        
        current_hdr_path = hdr_paths[current_hdr_index]
        current_hdr_name = current_hdr_path.split('/')[-1]
        name_wo_ext = current_hdr_name[:-4]

        # set up the environment map for hdr lighting
        mapping = build_environment_texture_background(world, current_hdr_path)
        print(f'[{i+1}/{len(hdr_indexes)}] Rendering using : {current_hdr_name} ')


        output_path_image_current = os.path.join(output_path_image, name_wo_ext)
        output_path_shading_current = os.path.join(output_path_shading, name_wo_ext)
        # create_or_recreate_folders(output_path_image_current)
        # create_or_recreate_folders(output_path_shading_current)

        # output_path_depth_current = os.path.join(output_path_depth, name_wo_ext)
        # output_path_mask_foreground_current = os.path.join(output_path_mask_foreground, name_wo_ext)
        # output_path_mask_background_current = os.path.join(output_path_mask_background, name_wo_ext)
        # output_path_background_current = os.path.join(output_path_background, name_wo_ext)
        # output_camera_path_current = os.path.join(output_camera_path, name_wo_ext)


    
    
        # ----------------------------------------------- image rendering ---------------------------------------------------
        render_images(scene, output_path_image_current, camera_object, num_samples, mapping, mesh, angles, name)

        # ------------------------------------ foreground shading rendering --------------------------------------
        # render_foreground_shading(scene, output_path_shading_current, camera_object, num_samples, mapping, mesh, angles, name)
    


    # -------------------------------------------- items that need single pass rendering--   --------------------------------

    # ----------------------------------------------- alpha foreground rendering -------------------------------------------
    # render_foreground_mask(scene, output_path_mask_foreground, camera_object, num_samples, mesh, rg)
    
    # ----------------------------------------------- alpha background rendering -------------------------------------------
    # render_background_mask(scene, output_path_mask_background, camera_object, num_samples, mesh, rg)
    
    # ----------------------------------------------- depth rendering ---------------------------------------------------

    # render_depth_map(scene, output_path_depth, camera_object, num_samples, mesh, rg)


    # ------------------------------------------------ renders background for test ---------------- -------------------------
    # render_background_image(camera_object, scene, output_path_background, num_samples, mapping)
    

    # ------------------------ save the camera intrinsic parameters ----------- -----------
    
#    cam_data = camera_object.data
#    K = get_calibration_matrix_K_from_blender(cam_data)
#    save_path = f'{output_camera_path}/camera.npy'
#    # print(K)
#    np.save(save_path, K)
    
    
main()
