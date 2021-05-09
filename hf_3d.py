import hou
import math
import numpy as np
from hausdorff import hausdorff_distance
from collections import defaultdict

node = hou.pwd()
geo = node.geometry()

lo_unclean_nodes = hou.session.find_nodes("oz_transform_input_")
merge_nodes = hou.session.find_nodes("oz_merge_")

point_boundaries = []
patch_point_boundaries = []
point_patchs = []
for point_group in geo.pointGroups():
  if "patch" in point_group.name():
    point_patchs.append(point_group)
  elif "boundary" in point_group.name():
    patch_point_boundaries.append(point_group)
  else:
    point_boundaries.append(point_group)

# ------------ Math Utility Functions ------------ #
def unord_hash(a, b):
  if a < b:
    return a * (b - 1) + math.trunc(math.pow(b - a - 2, 2)/ 4)
  elif a > b:
    return (a - 1) * b + math.trunc(math.pow(a - b - 2, 2)/ 4)
  else:
    return a * b + math.trunc(math.pow(abs(a - b) - 1, 2)/ 4)

# ------------ Geometry Utility Functions ------------ #
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

def best_fit_scale(lo_points_pos, hi_points_pos):
  best_scale, best_dist = 0.15, float('inf')
  min_scale, max_scale = 0, 1
  tries, max_tries = 0, 10
  sample_size = 50
  better_scale_findable = True
  while tries < max_tries and better_scale_findable:
    scales, dists = [], []
    for sample in range(sample_size):
      if sample == 0:
        scale = best_scale
      else:
        is_scale = False
        scale_tries, max_scale_tries = 0, 10
        while not is_scale and scale_tries < max_scale_tries:
          scale = np.random.normal(best_scale, (max_scale - min_scale) * 0.5/(tries+1))
          scale_tries += 1
          if scale >= min_scale and scale < max_scale:
            is_scale = True
        if not is_scale:
          scale = np.random.uniform(min_scale, max_scale)
      scales.append(scale)
      dists.append(hausdorff_distance(lo_points_pos, hi_points_pos * scale, 'manhattan'))


    min_dist, i = min((dist, i) for (i, dist) in enumerate(dists))
    if best_dist > min_dist:
      best_scale, best_dist = scales[i], min_dist
    else:
      better_scale_findable = False
  return best_scale, best_dist

class VirtualPolygon():
  # class to avoid generation of Houdini Polygons during intermediary phases
  def __init__(self, virtual, data):
    if virtual:
      self.ps = data
    else:
      ps = []
      for v in data.vertices():
        ps.append(v.point())
      self.ps = ps
    self.virtual = virtual

  def __eq__(self, other):
    same = True
    for p in self.ps:
      same = same and (p in other.ps)
    return same

  def __str__(self):
    string = []
    for p in self.ps:
      string.append(str(p.number()))
    string.sort()
    return "<" + ', '.join(string) + ">"

  def __repr__(self):
    return str(self)

  def get_edges(self):
    ps_zip1 = self.ps
    ps_zip2 = ps_zip1[1:] + [ps_zip1[0]]

    ps_zipped = []
    for p_zip1, p_zip2 in zip(ps_zip1, ps_zip2):
      if p_zip1.number() < p_zip2.number():
        ps_zipped.append([p_zip1, p_zip2])
      else:
        ps_zipped.append([p_zip2, p_zip1])
    return ps_zipped

  def get_common_edge(self, other):
    edges_self = self.get_edges()
    edges_other = other.get_edges()
    for edge_self in edges_self:
      if edge_self in edges_other:
        return edge_self
    return None

class MinTriangulation():
  def __init__(self, geo, points, cache_lengths=None):
    if cache_lengths is None:
      cache_lengths = defaultdict(list)
      for i in range(len(points)):
        for j in range(i+1, len(points)):
          p_i, p_j = points[i], points[j]
          pi_pos = p_i.position()
          pj_pos = p_j.position()
          cache_lengths[unord_hash(p_i.number(), p_j.number())] = (pi_pos - pj_pos).length()
    self.geo = geo
    self.points = points
    self.cache_lengths = cache_lengths
    self.cache_costs = defaultdict(list)

  def tri_cost(self, i, j, k, is_mwt=True):
    eik_len = self.cache_lengths[unord_hash(self.points[i].number(), self.points[k].number())]
    ekj_len = self.cache_lengths[unord_hash(self.points[k].number(), self.points[j].number())]
    if is_mwt:
      return eik_len + ekj_len
    else:
      eij_len = self.cache_lengths[unord_hash(self.points[i].number(), self.points[j].number())]
      s = eij_len + eik_len + ekj_len / 2
      return math.sqrt(s*(s-eij_len)*(s-eik_len)*(s-ekj_len))

  def tri_min(self, i, j):
    if (i, j) in self.cache_costs:
      return self.cache_costs[(i, j)]
    else:
      if j <= i+1:
        self.cache_costs[(i, j)] = (0, [])
        return (0, [])
      else:
        min_cost = float('inf')
        potential_polys = {}
        for k in range(i+1, j):
          cost_center = self.tri_cost(i, j, k)
          min_cost_left, min_polys_left = self.tri_min(i, k)
          min_cost_right, min_polys_right = self.tri_min(k, j)
          curr_cost = cost_center + min_cost_left + min_cost_right
          curr_polys = [VirtualPolygon(True, [self.points[i], self.points[j], self.points[k]])] + min_polys_left + min_polys_right
          if curr_cost < min_cost:
            min_cost = curr_cost
            potential_polys[curr_cost] = curr_polys
        min_polys = potential_polys[min_cost]
      self.cache_costs[(i, j)] = (min_cost, min_polys)
      return min_cost, min_polys

  def min_triangulation(self, generate=True):
    _, min_polys = self.tri_min(0, len(self.points)-1)
    if generate:
      for min_poly in min_polys:
        new_poly = self.geo.createPolygon()
        for p in min_poly.ps:
          new_poly.addVertex(p)
    return min_polys

# ------------ Hole-Filling Classes ------------ ~

class Island_Fill():
  def __init__(self, geo, points, inner_points):
    self.geo = geo
    self.points = points
    self.inner_points = inner_points

  def fill(self):
    '''
    Similar to the lo-frequency case, We Follow F Bi, Y Hu, X Chen, Y Ma [2013],
    Island hole automatic filling algorithm in triangular meshes
    A. Find the two points on the inner boundary furthest from one another
    '''
    inner_ps = self.inner_points
    max_dist, max_dist_ps = 0, None
    total_ps_inner = len(inner_ps)
    p_1_inner_ind, p_2_inner_ind = None, None
    for i, inner_p in enumerate(inner_ps):
      pos = inner_p.position()
      start_ind = (i + int(0.4 * total_ps_inner)) % total_ps_inner
      end_ind = (i + int(0.6 * total_ps_inner)) % total_ps_inner

      mid_ind = (total_ps_inner - 1) if end_ind < start_ind else end_ind
      for ind in range(start_ind, mid_ind):
        other_p = inner_ps[ind]
        curr_dist = (pos - other_p.position()).length()
        if curr_dist > max_dist:
          max_dist, max_dist_ps = curr_dist, (inner_p, other_p)
          p_1_inner_ind, p_2_inner_ind = i, ind

      mid_ind = 0 if end_ind < start_ind else end_ind
      for ind in range(mid_ind, end_ind):
        other_p = inner_ps[ind]
        curr_dist = (pos - other_p.position()).length()
        if curr_dist > max_dist:
          max_dist, max_dist_ps = curr_dist, (inner_p, other_p)
          p_1_inner_ind, p_2_inner_ind = i, ind

    '''
    B. Find each of the points, find the closest point on the outer boundary
    '''
    p_1_inner, p_2_inner = max_dist_ps[0], max_dist_ps[1]
    pos_1 = p_1_inner.position()
    pos_2 = p_2_inner.position()
    min_dist_1, min_dist_2 = float('inf'), float('inf')
    p_1_outer_ind, p_2_outer_ind = None, None
    p_1_outer, p_2_outer = None, None

    p_a, p_b = self.points[len(self.points)-1], self.points[1]
    p_1, p_2 = get_clockwise_neighbors(self.points[0], (p_a, p_b))
    outer_ps = self.points[::-1] if (p_a == p_1 and p_b == p_2) else self.points
    total_ps_outer = len(outer_ps)
    for i, point in enumerate(outer_ps):
      pos = point.position()
      if (pos - pos_1).length() < min_dist_1:
        p_1_outer_ind = i
        p_1_outer = point
        min_dist_1 = (pos - pos_1).length()
      if (pos - pos_2).length() < min_dist_2:
        p_2_outer_ind = i
        p_2_outer = point
        min_dist_2 = (pos - pos_2).length()
    
    '''
    C. Connect the inner points with their respective outer point.
        This forms two regular holes
    '''
    points_outer = outer_ps[p_1_outer_ind:] + outer_ps[:p_1_outer_ind]
    if p_1_outer_ind < p_2_outer_ind:
      p_2_outer_ind = p_2_outer_ind - p_1_outer_ind
    else:
      p_2_outer_ind = total_ps_outer - p_1_outer_ind + p_2_outer_ind
    p_1_outer_ind = 0

    points_inner = inner_ps[p_1_inner_ind:] + inner_ps[:p_1_inner_ind]
    if p_1_inner_ind < p_2_inner_ind:
      p_2_inner_ind = p_2_inner_ind - p_1_inner_ind
    else:
      p_2_inner_ind = total_ps_inner - p_1_inner_ind + p_2_inner_ind
    p_1_inner_ind = 0

    p_1_to_p_2_outer = np.append(np.array(points_outer[:p_2_outer_ind]), [p_2_outer])
    p_2_to_p_1_outer = np.append(np.array(points_outer[p_2_outer_ind:]), [p_1_outer])
    p_1_to_p_2_inner = np.append(np.array(points_inner[:p_2_inner_ind]), [p_2_inner])
    p_2_to_p_1_inner = np.append(np.array(points_inner[p_2_inner_ind:]), [p_1_inner])

    p_1_to_p_2_inner_rev = p_1_to_p_2_inner[::-1]
    p_2_to_p_1_inner_rev = p_2_to_p_1_inner[::-1]
    '''
    D. We can split the two regular holes to more holes, depending on the size of the hole
        Fill the holes using any method
    '''
    max_boundary_size = 200
    boundaries = [(p_1_to_p_2_outer, p_1_to_p_2_inner_rev), (p_2_to_p_1_outer, p_2_to_p_1_inner_rev)] # outer, inner
    num_split = 0
    while boundaries:
      print(len(boundaries))
      outer, inner = boundaries.pop()
      if len(outer) + len(inner) > max_boundary_size:
        inner_ind = int(math.floor(len(inner) / 2))
        inner_point = inner[inner_ind]
        inner_pos = inner_point.position()

        min_dist = float('inf')
        for i, point in enumerate(outer):
          pos = point.position()
          if (inner_pos - pos).length() < min_dist:
            outer_ind, outer_point = i, point
            outer_pos = outer_point.position()
            min_dist = (inner_pos - pos).length()
        inner_1 = np.append(inner[:inner_ind], [inner_point])
        inner_2 = inner[inner_ind:]
        outer_1 = outer[outer_ind:]
        outer_2 = np.append(outer[:outer_ind], [outer_point])
        boundaries.append((outer_1, inner_1))
        boundaries.append((outer_2, inner_2))
      else:
        num_split += 1
        MinTriangulation(self.geo, np.append(outer, inner)).min_triangulation(generate=True)


# ------------ Main Code ------------ #
for i, merge_node in enumerate(merge_nodes):
  points = point_boundaries[i].points()
  patch_points = patch_point_boundaries[i].points()
  points_patch = point_patchs[i].points()

  lo_unclean_node = lo_unclean_nodes[i]
  lo_node = merge_node.inputs()[0]
  hi_node = merge_node.inputs()[1]
  
  lo_unclean_points = lo_unclean_node.geometry().points()
  lo_points = lo_node.geometry().points()
  hi_points = hi_node.geometry().points()
  '''
  1. hi-freq patches are of different scale from original lo-freq patches
     Haudorff distance computes the "similarity" of the patches
     We can thus compute an ideal scale by an evolutionary algorithm on min Hausdorff distances
  '''
  lo_unclean_points_pos = []
  for lo_unclean_point in lo_unclean_points:
    lo_unclean_points_pos.append(lo_unclean_point.position())
  lo_unclean_points_pos = np.array(lo_unclean_points_pos)

  lo_points_pos = []
  for lo_point in lo_points:
    lo_points_pos.append(lo_point.position())
  lo_points_pos = np.array(lo_points_pos)

  hi_points_pos = []
  for hi_point in hi_points:
    hi_points_pos.append(hi_point.position())
  hi_points_pos = np.array(hi_points_pos)

  # NOTE: We compute hi patch similarity from aggregate of two best fits- against lo patch without boundary points and low patch with boundary points
  #       This theoretically creates a better fit, as hi_patch will neither be too close to actual boundary nor too small

  best_scale_1, best_dist_1 = best_fit_scale(lo_unclean_points_pos, hi_points_pos)
  best_scale_2, best_dist_2 = best_fit_scale(lo_points_pos, hi_points_pos)
  best_scale = (best_dist_2 * best_scale_1) / (best_dist_1 + best_dist_2) + (best_dist_1 * best_scale_2) / (best_dist_1 + best_dist_2)
  best_dist = (best_dist_2 * best_dist_1) / (best_dist_1 + best_dist_2) + (best_dist_1 * best_dist_2) / (best_dist_1 + best_dist_2)

  print("Scaled hi-freq patch by " + str(best_scale) + " to lo-freq patch size, with error " + str(best_dist))
  lo_node_translate = hou.Vector3((hou.session.find_parm(lo_unclean_node, "tx"), hou.session.find_parm(lo_unclean_node, "ty"), hou.session.find_parm(lo_unclean_node, "tz")))
  for point in points_patch:
    point.setPosition(point.position() * best_scale - lo_node_translate)
  
  '''
  2. We now have the original mesh and a hi-frequency patch mesh. This forms an island hole
     Repair via hole-stitching algorithms.
  '''
  Island_Fill(geo, points, patch_points).fill()

node.bypass(True)


