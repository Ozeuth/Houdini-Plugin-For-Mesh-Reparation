import hou
from collections import defaultdict

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
for_node = hou.node(hou.parent().path() + "/repeat_end")
'''
1. Remove tooth faces
tooth faces have > 1 boundary edges
'''
def get_tooth_faces(edges):
  polygon_to_edges = defaultdict(list)
  tooth_faces = []
  for edge in edges:
    for prim in edge.prims():
      if prim.type() == hou.primType.Polygon:
        polygon_to_edges[prim].append(edge)
  for key, value in polygon_to_edges.items():
    if len(value) > 1:
      tooth_faces.append(key)
  return tooth_faces

reval = False
edge_boundaries = inputs[1].geometry().edgeGroups()
for edge_boundary in edge_boundaries:
  tooth_faces = get_tooth_faces(edge_boundary.edges())
  if tooth_faces:
    geo.deletePrims(tooth_faces)
    reval = True
    
for_node.parm("stopcondition").set(int(not reval))