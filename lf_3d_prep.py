import hou
import math
import numpy as np
from collections import defaultdict

node = hou.pwd()
geo = node.geometry()

point_boundaries = []
neighbor_point_boundaries = []
for point_group in geo.pointGroups():
  if "neighbor" in point_group.name():
    neighbor_point_boundaries.append(point_group)
  else: 
    point_boundaries.append(point_group)

edge_boundaries = []
neighbor_edge_boundaries = []
for edge_group in geo.edgeGroups():
  if "neighbor" in edge_group.name():
    neighbor_edge_boundaries.append(edge_group)
  else:
    edge_boundaries.append(edge_group)

def get_poly(geo, ps):
  new_poly = geo.createPolygon()
  for p in ps:
    new_poly.addVertex(p)
  return new_poly

def get_clockwise_neighbors(p, p_a_b):
  # p_1 = left of p, p_2 = right of p
  p_a, p_b = p_a_b
  p_1, p_2 = None, None
  for prim in p.prims():
    if prim.type() == hou.primType.Polygon:
      ps = []
      for v in prim.vertices():
        ps.append(v.point())
        
      if p_a in ps or p_b in ps:
        p_i = ps.index(p)
        ps =  ps[p_i:] + ps[:p_i]
        assert(ps[0] == p)
        p_2 = ps[1] if (ps[1] == p_a or ps[1] == p_b) else p_2
        p_1 = ps[len(ps) - 1] if (ps[len(ps) - 1] == p_a or ps[len(ps) - 1] == p_b) else p_1
  assert(p_1 != None and p_2 != None)
  return p_1, p_2

temp_polys = []
for i in range(1, len(point_boundaries)):
  points = point_boundaries[i].points()
  temp_poly = get_poly(geo, points)
  temp_polys.append(temp_poly)

for i in range(1, len(point_boundaries)):
  '''
  Compute Projection Plane via Least Squares

    Plane eqn: ax + by + c = z
        A        x   =   b
    | x0 y0 1 |         | z0 |
    | x1 y1 1 | | a |   | z1 |  => | a |
    | ....... | | b | = | .. |     | b | = (A^T*A)^-1*A^T*b
    | xn yn 1 | | c |   | zn |     | c |
  '''
  points = point_boundaries[i].points()
  edges = edge_boundaries[i].edges()
  A = []
  b = []
  boundary_center = np.array([0, 0, 0])
  boundary_normal = hou.Vector3((0, 0, 0))
  points_neighbors = defaultdict(list)
  for edge in edges:
    p_1, p_2 = edge.points()
    points_neighbors[p_1].append(p_2)
    points_neighbors[p_2].append(p_1)
  for point in points:
    point_pos = point.position()
    A.append([point_pos[0], point_pos[1], 1])
    b.append(point_pos[2])
    boundary_center = boundary_center + np.array(point_pos)
    p_l, p_r = get_clockwise_neighbors(point, points_neighbors[point])
    e_dir1 = hou.Vector3(p_l.position() - point.position()).normalized()
    e_dir2 = hou.Vector3(p_r.position() - point.position()).normalized()
    point_normal = e_dir2.cross(e_dir1).normalized()
    boundary_normal += point_normal
  boundary_normal /= len(points)
  A = np.matrix(A)
  b = np.matrix(b).T
  boundary_center /= len(points)
  fit_fail = False
  det = np.linalg.det(A.T * A)
  if (det > 0.000001):
    fit = (A.T * A).I * A.T * b
  else:
    # This plane is almost parallel to xz plane
    fit = np.linalg.pinv(A.T * A) * A.T * b
    fit_fail = True
  a = fit.item(0)
  b = fit.item(1)
  c = fit.item(2)
  errors = b - A * fit
  residual = np.linalg.norm(errors)

  if (not fit_fail):
    plane_normal = np.array([a, b, -1]) / math.sqrt(math.pow(a, 2) + math.pow(b, 2) + 1)
  else:
    plane_normal = np.array([0.001, 1, 0.001])* (1 if (a >= 0) else -1)

  hit_positions = []
  prev_dist = 0.01
  ray_origin, ray_direction = hou.Vector3(boundary_center) + 0.1 * hou.Vector3(plane_normal), hou.Vector3(-1 * plane_normal)
  inter_p, inter_n, inter_uvw= hou.Vector3(), hou.Vector3(), hou.Vector3()

  inter = geo.intersect(ray_origin, ray_direction, inter_p, inter_n, inter_uvw, min_hit=prev_dist)
  while inter != -1:
    hit_positions.append(hou.Vector3(inter_p))
    prev_dist = ray_origin.distanceTo(inter_p) + 0.01
    inter = geo.intersect(ray_origin, ray_direction, inter_p, inter_n, inter_uvw, min_hit=prev_dist)
  num_hits = len(hit_positions)

  plane_normal = hou.Vector3(plane_normal) if num_hits % 2 == 0 else hou.Vector3(-1 * plane_normal)
  if not geo.findGlobalAttrib("normal_" + str(i)):
    geo.addAttrib(hou.attribType.Global, "normal_" + str(i), plane_normal)
  else:
    geo.setGlobalAttribValue("normal_" + str(i), plane_normal)
geo.deletePrims(temp_polys, keep_points=True)
node.bypass(True)

