import hou
import numpy as np

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
boundaries = inputs[1].geometry().pointGroups()
edge_boundaries = inputs[2].geometry().edgeGroups()
'''
2. Fill small holes with centroid
                     | 1 if h has <= 8 points in it
  is_small_hole(h) = | 0 otherwise
'''
for i in range(1, len(boundaries)):
  points = boundaries[i].points()
  if len(points) <= 8:
    edges = edge_boundaries[i-1].edges()
    center = np.zeros(3)
    normal = np.zeros(3)
    for point in points:
      center += point.position()
      normal += point.attribValue("N")
    center /= len(points)
    normal /= len(points)
    centroid = geo.createPoint()
    centroid.setPosition(center)
    centroid.setAttribValue("N", normal)
    for edge in edges:
      new_poly = geo.createPolygon()
      new_poly.addVertex(centroid)
      edge_points = edge.points()
      for edge_point in edge_points:
        new_poly.addVertex(edge_point)
node.bypass(True)