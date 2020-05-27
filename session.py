# -- Houdini Mesh Repairer -- #
import hou
import nodesearch
import os
import time

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
    if hou.node(node.path() + "/prepare_2d"):
      node_2d = hou.node(node.path() + "/prepare_2d")
  assert (node_3d and node_2d), ("ERROR: Please reinstate Digital Asset")
  node_3d.bypass(False)
  '''
  4. Generate 2D Render
  '''
  image_paths = []
  renders = hou.nodeType(hou.ropNodeTypeCategory(), "ifd").instances()
  for render in renders:
    if "oz_render_" in render.name():
      image_paths.append(render.parm("vm_picture").eval())
      render.render()
  '''
  5. Map 3D Context Region -> 2D Render
  '''
  for image_path in image_paths:
    while (os.path.isfile(image_path + ".mantra_checkpoint")):
      time.sleep(1)
  node_2d.bypass(False)