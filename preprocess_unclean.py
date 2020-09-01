import hou
from collections import defaultdict

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
for_node = hou.node(hou.parent().path() + "/repeat_end")
smooth_node = hou.node(hou.parent().path() + "/switch")
'''
We follow C Feng, J Liang, M Ren, G Qiao, W Lu, S Li [2020],
  infilling using additive repair
  Special Thanks to:
    François Chollet for his np-only minimizer implementation
    Jenny Zheng for work on the improved AFT

1. Remove tooth faces
                     | 1 if f is n-gon with n-1 boundary edges
  is_tooth_face(f) = | 0 otherwise
'''
def get_tooth_faces(edges):
  polygon_to_edges = defaultdict(list)
  tooth_faces = []
  for edge in edges:
    for prim in edge.prims():
      if prim.type() == hou.primType.Polygon:
        polygon_to_edges[prim].append(edge)
  for key, value in polygon_to_edges.items():
    if len(value) >= len(key.points()) - 1:
      tooth_faces.append(key) 
  return tooth_faces

stop = True
edge_boundaries = inputs[1].geometry().edgeGroups()
for edge_boundary in edge_boundaries:
  tooth_faces = get_tooth_faces(edge_boundary.edges())
  if tooth_faces:
    geo.deletePrims(tooth_faces)
    stop = False

prior_stop = geo.findGlobalAttrib("stop") if geo.findGlobalAttrib("stop") != None else False
stop = stop or prior_stop
smooth_node.parm("input").set(int(stop))
if stop:
  for_node.parm("stopcondition").set(int(stop))
  geo.addAttrib(hou.attribType.Global, "stop", True)