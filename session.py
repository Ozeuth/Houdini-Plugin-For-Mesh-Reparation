# -- Houdini Mesh Repairer -- #
import hou
import glob
import nodesearch
import os
from PIL import ImageFont
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

def find_parm(node, name):
  params = hou.parent().parms()
  found_eval = None
  for param in params:
    if (name in param.name()):
      found_eval = param.eval()
      break
  return found_eval

def reset_camera(camera):
  camera.parmTuple('t').set((0, 0, 0))
  camera.parmTuple('r').set((0, 0, 0))

def reset_camera_info():
  cameras_info["center"] = []
  cameras_info["rotationsx"] = []
  cameras_info["rotationsy"] = []
  cameras_info["normals"] = []
  cameras_info["zooms"] = []

def render_then_map(image_paths, node_2d):
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

def low_repair():
  # Low Frequency Pass
  matcher = nodesearch.Name(HDA_name)
  for node in matcher.nodes(hou.node("/obj/"), recursive=True):
    if hou.node(node.path() + "/repeat_end"):
      node_prep = hou.node(node.path() + "/repeat_end")
    if hou.node(node.path() + "/lf_3d"):
      node_prep_clean = hou.node(node.path() + "/lf_3d")
  '''
  1-2. Clean Tooth Edges
       Smooth Boundaries
  '''
  node_prep.parm("stopcondition").set(0)
  '''
  3. Topology Repair
  '''
  node_prep_clean.bypass(False)

def high_repair():
  # High Frequency Pass
  matcher = nodesearch.Name(HDA_name)
  for node in matcher.nodes(hou.node("/obj/"), recursive=True):
    if hou.node(node.path() + "/hf_prepare_3d"):
      node_3d = hou.node(node.path() + "/hf_prepare_3d")
    if hou.node(node.path() + "/hf_optimize_3d"):
      node_op_3d = hou.node(node.path() + "/hf_optimize_3d")
    if hou.node(node.path() + "/hf_prepare_2d"):
      node_2d = hou.node(node.path() + "/hf_prepare_2d")
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
  num_images = len(glob.glob(hou.hipFile.name().split(".")[0] + "/*.png"))
  mark_for_destroy = []
  image_paths = []
  cameras = hou.nodeType(hou.objNodeTypeCategory(),"cam").instances()
  for camera in cameras:
    camera_name = camera.name()
    if "oz_camera_" in camera_name and int(filter(str.isdigit, camera_name)) > num_images:
      mark_for_destroy.append(camera)
  renders = hou.nodeType(hou.ropNodeTypeCategory(), "ifd").instances()
  for render in renders:
    render_name = render.name()
    if "oz_render_" in render_name:
      if int(filter(str.isdigit, render_name)) <= num_images:
        image_paths.append(render.parm("vm_picture").eval())
        render.render()
      else:
        mark_for_destroy.append(render)
  '''
  6. Map 3D Region -> 2D Render
  7. 2D Repair
  '''
  #render_then_map(image_paths, node_2d)
  render_map_thread = Thread(target=render_then_map, args=(image_paths,node_2d,))
  render_map_thread.start()
  '''
  *. Clean-Up
  '''
  for node in mark_for_destroy:
    node.destroy()
  reset_camera_info()

# Both Passes
def repair():
  low_repair()
  high_repair()