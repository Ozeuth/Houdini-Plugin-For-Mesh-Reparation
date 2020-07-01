import hou
import numpy as np

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
boundaries = inputs[1].geometry().pointGroups()
edge_boundaries = inputs[2].geometry().edgeGroups()
'''
2. Fill small holes with centroid
                     | 1 if h 
  is_small_hole(h) = | 0 otherwise
'''
for i in range(1, len(boundaries)):
  points = boundaries[i].points()
  edges = edge_boundaries[i-1].edges()
  center = np.zeros(3)
  for point in points:
    center += point.position()
  center /= len(points)
  if len(points) <= 8:
    centroid = geo.createPoint()
    centroid.setPosition(center)
    # sort by x then z?
    for edge in edges:
      new_poly = geo.createPolygon()
      new_poly.addVertex(centroid)
      edge_points = edge.points()
      for edge_point in edge_points:
        new_poly.addVertex(edge_point)
