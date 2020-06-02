# -- Houdini Mesh Repairer -- #
import glob
import hou
import nodesearch
import os
from PIL import ImageFont
import time
from threading import Thread

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

def repair():
  node_3d = None
  node_2d = None
  '''
  1-3: Choose 3D Context Region
  '''
  matcher = nodesearch.Name("mesh_repairer")
  for node in matcher.nodes(hou.node("/obj/"), recursive=True):
    if hou.node(node.path() + "/prepare_3d"):
      node_3d = hou.node(node.path() + "/prepare_3d")
    if hou.node(node.path() + "/optimize_3d"):
      node_op_3d = hou.node(node.path() + "/optimize_3d")
    if hou.node(node.path() + "/prepare_2d"):
      node_2d = hou.node(node.path() + "/prepare_2d")
  assert (node_3d and node_op_3d and node_2d), ("ERROR: Please reinstate Digital Asset")
  node_3d.bypass(False)
  node_op_3d.bypass(False)
  '''
  6. Generate 2D Render
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
  7. Map 3D Context Region -> 2D Render
  '''
  #render_then_map(image_paths, node_2d)
  render_map_thread = Thread(target=render_then_map, args=(image_paths,node_2d,))
  render_map_thread.start()
  '''
  *. Clean-Up
  '''
  for node in mark_for_destroy:
    node.destroy()