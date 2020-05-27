# -- Houdini Mesh Repairer -- #
import hou
import nodesearch

def repair():
  '''
  1-3: Choose 3D Context Region
  '''
  matcher = nodesearch.Name("mesh_repairer")
  for node in matcher.nodes(hou.node("/obj/"), recursive=True):
    if hou.node(node.path() + "/prepare_3d"):
      hou.node(node.path() + "/prepare_3d").bypass(False)
  '''
  4. Generate 2D Render
  '''
  renders = hou.nodeType(hou.ropNodeTypeCategory(), "ifd").instances()
  for render in renders:
    if "oz_render_" in render.name():
      render.render()

  '''
  5. Map 3D Context Region -> 2D Render
  '''