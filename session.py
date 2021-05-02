# -- Houdini Mesh Repairer -- #
import hou
import glob
import nodesearch
import os
from PIL import ImageFont
import subprocess as sp
import time
from threading import Thread

HDA_name = "Mesh_Repairer_Oz"
cameras_info = {
  "centers": [],
  "rotationsx": [],
  "rotationsy": [],
  "normals": [],
  "zooms": []
}
# ------------ Generic Utility Functions ------------ #
def find_parm(node, name):
  params = node.parms()
  found_eval = None
  for param in params:
    if (name in param.name()):
      found_eval = param.eval()
      break
  return found_eval


def find_nodes(name, num_nodes=float('inf'), in_hda=True):
  matcher = nodesearch.Name(name)
  matching_nodes = []
  found_nodes = 0
  for node in matcher.nodes(hou.node("/obj/"), recursive=True):
    if (in_hda and (HDA_name in node.path())) or not in_hda:
      matching_nodes.append(node)
      found_nodes += 1
      if found_nodes == num_nodes:
        break
  return matching_nodes[0] if num_nodes == 1 else matching_nodes

# ------------ Pipeline Functions ------------ #
def reset_camera(camera):
  camera.parmTuple('t').set((0, 0, 0))
  camera.parmTuple('r').set((0, 0, 0))

def reset_camera_info():
  cameras_info["center"] = []
  cameras_info["rotationsx"] = []
  cameras_info["rotationsy"] = []
  cameras_info["normals"] = []
  cameras_info["zooms"] = []

def render_then_map(image_paths, node_2d, is_full):
  '''
  NOTE: This is a tenuous solution to a synchronicity issue.
  Absence of a rendering checkpoint file means:
    A. Rendering has ended
    OR
    B. Rendering has not started
  We must first check that all renders have started, then that all renders have ended
  This must be done in a thread (Or renders that have not started will not start)
  '''
  print("Rendering Start!")
  render_started = [False] * len(image_paths)
  while sum(render_started) != len(image_paths):
    for i in range(len(image_paths)):
      render_started[i] = render_started[i] or os.path.isfile(image_paths[i] + ".mantra_checkpoint")
    time.sleep(1)

  for image_path in image_paths:
    while (os.path.isfile(image_path + ".mantra_checkpoint")):
      time.sleep(1)
  print("Rendering Complete!")
  node_2d.bypass(False)
  if is_full:
    low_repair()

def preprocess():
  node_prep = find_nodes("repeat_end", num_nodes=1)
  '''
  0. Clean Tooth Edges
  '''
  node_prep.parm("stopcondition").set(0)

def low_repair():
  preprocess()
  # Low Frequency Pass
  node_lf = find_nodes("lf_3d", num_nodes=1)
  '''
  1. Topology Repair
  '''
  node_lf.bypass(False)

def high_repair():
  global process
  # High Frequency Pass
  node_hf_prep = find_nodes("hf_3d_prep", num_nodes=1)
  '''
  2. Export Patch Topology
  '''
  node_hf_prep.bypass(False)
  '''
  3. Detail Repair
  '''
  synthesizer_path = find_parm(find_nodes(HDA_name, num_nodes=1, in_hda=False), "synth_path")
  # NOTE: Houdini parm template ignores the first / in path
  if not os.path.exists(synthesizer_path):
    if os.path.exists("/" + synthesizer_path):
      synthesizer_path = "/" + synthesizer_path
    else:
      raise Exception("ERROR: Path to geometric texture synthesizer invalid")

  process = sp.Popen("wsl /home/ozeuth/anaconda3/envs/test/bin/python /home/ozeuth/geometric-textures/repair.py cloud patch_group__0",
                              stdout=sp.PIPE, shell=True)
  while True:
    line = process.stdout.readline()
    if not line: 
      break
    print(line, flush=True)
  process.stdout.close()
  return_code = process.wait()
  if return_code:
    raise sp.CalledProcessError(return_code, "geometric synthesizer failed")

  merge_node_ = find_nodes("oz_combined", num_nodes=1)
  merge_node_.cook()
  input_transform_nodes = find_nodes("oz_transform_input_")
  file_output_nodes = find_nodes("oz_output_")
  output_transform_nodes = find_nodes("oz_transform_output_")
  merge_nodes = find_nodes("oz_merge_")

  for i in range(len(input_transform_nodes)):
    input_transform_node = input_transform_nodes[i]
    file_output_node = file_output_nodes[i]
    output_transform_nodes = output_transform_nodes[i]
    merge_node = merge_nodes[i]
    '''
    4. Import Detail Repaired Patch
    ''' 
    file_output_node.parm("reload").pressButton()
    '''
    5. Align Detail Repaired Patch
    '''
    input_transform_node.parm("movecentroid").pressButton()
    output_transform_nodes.parm("movecentroid").pressButton()
    merge_node_.setInput(i+2, merge_node)
    
    

def high_repair_old(is_full=False):
  preprocess()
  # High Frequency Pass
  node_3d = find_nodes("hf_prepare_3d", num_nodes=1)
  node_op_3d = find_nodes("hf_optimize_3d", num_nodes=1)
  node_2d = find_nodes("hf_prepare_2d", num_nodes=1)
  '''
  1. Choose 3D Context Region
  '''
  node_3d.bypass(False)
  '''
  2-4. 3D Context Region Optimization
  '''
  node_op_3d.bypass(False)
  '''
  5. Generate 2D Render
  '''
  num_images = len(glob.glob(hou.hipFile.name().split(".")[0] + "/*_opening.png"))
  mark_for_destroy = []
  image_paths = []
  cameras = hou.nodeType(hou.objNodeTypeCategory(),"cam").instances()
  for camera in cameras:
    camera_name = camera.name()
    if "oz_camera_" in camera_name and int("".join(filter(str.isdigit, camera_name))) > num_images:
      mark_for_destroy.append(camera)
  renders = hou.nodeType(hou.ropNodeTypeCategory(), "ifd").instances()
  for render in renders:
    render_name = render.name()
    if "oz_render_" in render_name:
      if int("".join(filter(str.isdigit, render_name))) <= num_images:
        image_paths.append(render.parm("vm_picture").eval())
        render.render()
      else:
        mark_for_destroy.append(render)
  '''
  6. Map 3D Region -> 2D Render
  7. 2D Repair
  '''
  #render_then_map(image_paths, node_2d)
  render_map_thread = Thread(target=render_then_map, args=(image_paths,node_2d,is_full,))
  render_map_thread.start()
  '''
  *. Clean-Up
  '''
  for node in mark_for_destroy:
    node.destroy()
  reset_camera_info()

# Both Passes
def repair():
  preprocess()
  high_repair(is_full=True)