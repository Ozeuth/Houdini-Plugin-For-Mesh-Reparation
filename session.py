# -- Houdini Mesh Repairer -- #
import glob
import hou
import nodesearch
import os
import time

def repair():
  node_3d = None
  node_2d = None
  '''
  1-3: Init and Choose 3D Context Region
  '''
  matcher = nodesearch.Name("mesh_repairer")
  for node in matcher.nodes(hou.node("/obj/"), recursive=True):
    if hou.node(node.path() + "/prepare_3d"):
      node_3d = hou.node(node.path() + "/prepare_3d")
    if hou.node(node.path() + "/prepare_2d"):
      node_2d = hou.node(node.path() + "/prepare_2d")
  assert (node_3d and node_2d), ("ERROR: Please reinstate Digital Asset")
  node_3d.bypass(False)
  '''
  4. Generate 2D Render
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
  5. Map 3D Context Region -> 2D Render
  '''
  for image_path in image_paths:
    while (os.path.isfile(image_path + ".mantra_checkpoint")):
      time.sleep(1)
  node_2d.bypass(False)

  # Clean up
  for node in mark_for_destroy:
    node.destroy()