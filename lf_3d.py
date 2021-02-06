import hou
import itertools
import math
import numpy as np
import operator
import random
import copy
from collections import defaultdict
from PIL import Image, ImageDraw
import time

is_debug = True
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

# ------------ Math Utility Functions ------------ #
def unord_hash(a, b):
  if a < b:
    return a * (b - 1) + math.trunc(math.pow(b - a - 2, 2)/ 4)
  elif a > b:
    return (a - 1) * b + math.trunc(math.pow(a - b - 2, 2)/ 4)
  else:
    return a * b + math.trunc(math.pow(abs(a - b) - 1, 2)/ 4)

def lss(A, b):
  num_vars = A.shape[1]
  rank = np.linalg.matrix_rank(A)
  if rank == num_vars:
    sol = np.linalg.lstsq(A, b)[0]
    return (sol, True)
  else:
    sols = []
    for nz in itertools.combinations(range(num_vars), rank):
      try:
        sol = np.zeros((num_vars, 1))
        sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
        sols.append(sol)
      except np.linalg.LinAlgError:
        pass
    return (sols, False)

# NOTE: Houdini 2020 does not support scipy, using FC's np-only minimizer
def minimize(f, x_start, step=0.1, no_improve_thr=10e-6, no_improv_break=10,
             max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5):
  dim = len(x_start)
  prev_best = f(x_start)
  no_improv = 0
  res = [[x_start, prev_best]]

  for i in range(dim):
    x = copy.copy(x_start)
    x[i] = x[i] + step
    score = f(x)
    res.append([x, score])

  iters = 0
  while 1:
    res.sort(key=lambda x: x[1])
    best = res[0][1]

    if max_iter and iters >= max_iter:
      return res[0]
    iters += 1

    if best < prev_best - no_improve_thr:
      no_improv = 0
      prev_best = best
    else:
      no_improv += 1

    if no_improv >= no_improv_break:
      return res[0]

    x0 = [0.] * dim
    for tup in res[:-1]:
      for i, c in enumerate(tup[0]):
        x0[i] += c / (len(res)-1)

    xr = x0 + alpha*(x0 - res[-1][0])
    rscore = f(xr)
    if res[0][1] <= rscore < res[-2][1]:
      del res[-1]
      res.append([xr, rscore])
      continue

    if rscore < res[0][1]:
      xe = x0 + gamma*(x0 - res[-1][0])
      escore = f(xe)
      if escore < rscore:
        del res[-1]
        res.append([xe, escore])
        continue
      else:
        del res[-1]
        res.append([xr, rscore])
        continue

    xc = x0 + rho*(x0 - res[-1][0])
    cscore = f(xc)
    if cscore < res[-1][1]:
      del res[-1]
      res.append([xc, cscore])
      continue

    x1 = res[0][0]
    nres = []
    for tup in res:
      redx = x1 + sigma*(tup[0] - x1)
      score = f(redx)
      nres.append([redx, score])
    res = nres

# ------------ Geometry Utility Functions ------------ #
def get_poly(geo, ps):
  new_poly = geo.createPolygon()
  for p in ps:
    new_poly.addVertex(p)
  return new_poly

def get_normal(geo, p):
  total_w = 0
  ws, ns = [], []
  for prim in p.prims():
    if prim.type() == hou.primType.Polygon:
      w, n = prim.intrinsicValue("measuredarea"), prim.normal()
      total_w += w
      ws.append(w)
      ns.append(n)
  normal = hou.Vector3(0, 0, 0)
  try:
    for w, n in zip(ws, ns):
      normal += w / total_w * n
  except:
    for w, n in zip(ws, ns):
      normal += w * n
    normal = normal.normalized()
  p.setAttribValue("N", normal)
  return normal

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

# ------------ Hole-Filling Classes ------------ #
class Centroid_Fill():
  def __init__(self, geo, points, edges):
    self.geo = geo
    self.points = points
    self.edges = edges

  def fill(self):
    center = np.zeros(3)
    normal = np.zeros(3)
    for point in self.points:
      center += point.position()
      normal += point.attribValue("N")
    center /= len(self.points)
    normal /= len(self.points)
    centroid = self.geo.createPoint()
    centroid.setPosition(center)
    centroid.setAttribValue("N", normal)
    for edge in self.edges:
      ps = list(edge.points()) + [centroid]
      get_poly(self.geo, ps)

class Projection_BiLaplacian_Fill():
  def __init__(self, geo, points, edges, edges_neighbors):
    self.geo = geo
    self.points = points
    self.edges = edges
    self.edges_neighbors = edges_neighbors

  def find_circle(self, pA, pB, pC):
    # Find center in 2D (u, v) space, then project back to 3D space
    A = pA.position()
    B = pB.position()
    C = pC.position()
    u1 = B - A
    w1 = (C - A).cross(u1)
    u = u1 / u1.length()
    w = w1 / w1.length()
    v = w.cross(u)

    b = (u1.dot(u) , 0)
    c = ((C - A).dot(u), (C - A).dot(v))
    h = (math.pow((c[0]-b[0])/2, 2) + math.pow(c[1], 2) - math.pow(b[0]/2, 2)) / (2 * c[1])
    center = A + (b[0] / 2) * u + h * v
    radius = max((A - center).length(), max((C - center).length(), (C - center).length()))
    return center, radius

  def fill(self):
    '''
    A. Initialize with minimum area triangulation
    '''
    e_lens_hashed = {}
    for i in range(len(self.points)):
      for j in range(i+1, len(self.points)):
        p_i, p_j = self.points[i], self.points[j]
        pi_pos = p_i.position()
        pj_pos = p_j.position()
        e_lens_hashed[unord_hash(p_i.number(), p_j.number())] = (pi_pos - pj_pos).length()
    min_polys = MinTriangulation(self.geo, self.points, cache_lengths=e_lens_hashed).min_triangulation(generate=False)
    '''
    B. Conduct Triangle Splitting
      We split the minimum polygons with centroid-based method if:
      ALL_t, t elem(p_i, p_j, p_k), sqrt(2) * ||p_c-t|| > s(p_c) and sqrt(2) * ||p_c-t|| > s(t)
      where
        p_i, p_j, p_k = points of minimum polygon
        p_c = center point of minimum polygon
        s = scale factor function, average length of edges connected to point except
            for hole boundary edges
    '''
    points_neighbors = defaultdict(list)
    exterior_points = []
    exterior_edges_neighbors = defaultdict(list)
    for edges_neighbor in list(set(self.edges_neighbors) - set(self.edges)):
      p_1, p_2 = edges_neighbor.points()
      points_neighbors[p_1].append(p_2)
      points_neighbors[p_2].append(p_1)
      p1_pos = p_1.position()
      p2_pos = p_2.position()
      e_lens_hashed[unord_hash(p_1.number(), p_2.number())] = (p1_pos - p2_pos).length()
      for prim in edges_neighbor.prims():
        if prim.type() == hou.primType.Polygon:
          exterior_edges_neighbors[unord_hash(p_1.number(), p_2.number())].append(VirtualPolygon(False, prim))
      exterior_points.append(p_1) if p_1 not in exterior_points else exterior_points
      exterior_points.append(p_2) if p_2 not in exterior_points else exterior_points

    exterior_edges_hashed = []
    for edge in self.edges:
      p_1, p_2 = edge.points()
      exterior_edges_hashed.append(unord_hash(p_1.number(), p_2.number()))

    interior_edges_neighbors = defaultdict(list)
    for min_poly in min_polys:
      for p_1, p_2 in min_poly.get_edges():
        if not unord_hash(p_1.number(), p_2.number()) in exterior_edges_hashed:
          interior_edges_neighbors[unord_hash(p_1.number(), p_2.number())].append(min_poly)
          points_neighbors[p_2].append(p_1)
          points_neighbors[p_1].append(p_2)

    new_min_polys = min_polys
    min_polys_created = True
    while min_polys_created:
      min_polys_created = False
      fixed_new_min_polys = new_min_polys
      for min_poly in fixed_new_min_polys:
        p_i, p_j, p_k = min_poly.ps
        ts = [p_i, p_j, p_k]
        center = (p_i.position() + p_j.position() + p_k.position()) / 3
        e_lens = [(center - p_i.position()).length(), (center - p_j.position()).length(), (center - p_k.position()).length()]
        c_scale = sum(e_lens)
        c_normal = np.zeros(3)

        split = True
        for t in ts:
          c_normal += t.attribValue("N")
          t_scale = 0
          t_neighbors = points_neighbors[t]
          for t_neighbor in t_neighbors:
            t_scale += e_lens_hashed[unord_hash(t.number(), t_neighbor.number())]
          if math.sqrt(2) * (center - t.position()).length() <= min(t_scale, c_scale):
            split = False
        c_normal /= 3

        if split:
          p_c = self.geo.createPoint()
          p_c.setPosition(center)
          p_c.setAttribValue("N", c_normal)
          new_min_polys.remove(min_poly)
          new_min_polys.extend([VirtualPolygon(True, [p_i, p_c, p_j]), VirtualPolygon(True, [p_i, p_c, p_k]), VirtualPolygon(True, [p_k, p_c, p_j])])
          for t in ts:
            e_lens_hashed[unord_hash(t.number(), p_c.number())] = e_lens.pop()
            points_neighbors[t].append(p_c)
            others = list(filter(lambda x: x != t, ts))
            interior_edges_neighbors[unord_hash(t.number(), p_c.number())] = [VirtualPolygon(True, [t, p_c, others[0]]), VirtualPolygon(True, [t, p_c, others[1]])]
          points_neighbors[p_c] = ts
          for t_1, t_2 in min_poly.get_edges():
            if not unord_hash(t_1.number(), t_2.number()) in exterior_edges_hashed:
              interior_edges_neighbors[unord_hash(t_1.number(), t_2.number())].remove(min_poly)
              interior_edges_neighbors[unord_hash(t_1.number(), t_2.number())].append(VirtualPolygon(True, [t_1, p_c, t_2]))
          min_polys_created = True
    '''
    C. Conduct Edge-Swapping
      For two new polygons (p_i, p_j, p_k), (p_i, p_j, p_m) adjacent to interior edge eij,
                                | 1 p_k lies outside circumsphere created with p_i, p_j, p_m
                                |   OR p_m lies outside cirumsphere created with p_i, p_j, p_k
      is_locally_delaunay(ts) = | 0 otherwise
      if two polygons are not locally delaunay, swap edges. Replace eij with ekm.
    '''
    marked_for_update = {}
    marked_for_deletion = []
    for interior_edge in interior_edges_neighbors:
      poly_1, poly_2 = interior_edges_neighbors[interior_edge]
      common_edge = poly_1.get_common_edge(poly_2)
      poly_1_p = list(set(poly_1.ps) - set(common_edge))[0]
      poly_2_p = list(set(poly_2.ps) - set(common_edge))[0]

      poly_1_circumsphere_c, poly_1_circumsphere_r = self.find_circle(common_edge[0], poly_1_p, common_edge[1])

      if (poly_1_circumsphere_c - poly_2_p.position()).length() < poly_1_circumsphere_r:
        new_poly_1 = VirtualPolygon(True, [poly_1_p, common_edge[0], poly_2_p])
        new_poly_2 = VirtualPolygon(True, [poly_1_p, common_edge[1], poly_2_p])
        new_min_polys.remove(poly_1)
        new_min_polys.remove(poly_2)
        new_min_polys.extend([new_poly_1, new_poly_2])

        marked_for_deletion.append(interior_edge)
        marked_for_update[unord_hash(poly_1_p.number(), poly_2_p.number())] = [new_poly_1, new_poly_2]
        # 4 interior_edges_neighbors could potentially need updating
        e1, e2, e3, e4 = (unord_hash(poly_1_p.number(), common_edge[0].number()),
                          unord_hash(poly_1_p.number(), common_edge[1].number()),
                          unord_hash(poly_2_p.number(), common_edge[0].number()),
                          unord_hash(poly_2_p.number(), common_edge[1].number()))
        old_ps = [poly_1, poly_1, poly_2, poly_2]
        new_ps = [new_poly_1, new_poly_2, new_poly_1, new_poly_2]
        for e_i, e in enumerate([e1, e2, e3, e4]):
          if ((e in interior_edges_neighbors and not e in marked_for_deletion) or (e in marked_for_update)):
            update = interior_edges_neighbors if (e in interior_edges_neighbors and not e in marked_for_deletion) else marked_for_update
            update[e].remove(old_ps[e_i])
            update[e].append(new_ps[e_i])

        points_neighbors[common_edge[0]].remove(common_edge[1])
        points_neighbors[common_edge[1]].remove(common_edge[0])
        points_neighbors[poly_1_p].append(poly_2_p)
        points_neighbors[poly_2_p].append(poly_1_p)
    for marked in marked_for_deletion:
      del interior_edges_neighbors[marked]
    interior_edges_neighbors.update(marked_for_update)

    for min_poly in new_min_polys:
      get_poly(self.geo, min_poly.ps)

    '''
    D. Conduct Patch Fairing.
    Compute the Laplace Beltrami Matrix,
        pi  pj  pk
    pi | Li  Wij Wik |
    pj | Wji Lj  Wjk |
    pk | Wki Wkj Lk  |
    where
      Li = | Wij + Wik + ... if pi is a generated point
           | 1               otherwise
            | 0.5*(cot(alpha) + cot(beta)) if eij
      Wij = | 0                            otherwise
      where for two polygons (p_i, p_j, p_k), (p_i, p_j, p_m) adjacent to interior edge eij,
        alpha = angle_ikj
        beta = angle_imj
    Then solve f, f = [f0, f1, f2]
    M *    fd    =    vd
    M * | fd_i | = | vd_i |
        | fd_j |   | vd_j |
        | fd_k |   | vd_k |
    where
             | pi_pos[d]  if pi is not a generated point
      vd_i = | 0          otherwise
    '''    
    for edge in self.edges:
      p_1, p_2 = edge.points()
      for prim in edge.prims():
        if prim.type() == hou.primType.Polygon:
          exterior_edges_neighbors[unord_hash(p_1.number(), p_2.number())].append(VirtualPolygon(False, prim))

    hole_edges_neighbors = {}
    hole_edges_neighbors.update(interior_edges_neighbors)
    hole_edges_neighbors.update(exterior_edges_neighbors)
    laplace_beltrami = np.zeros((len(points_neighbors), len(points_neighbors)))
    laplace_vs = np.zeros((len(points_neighbors), 3))
    ref_keys = list(points_neighbors.keys())
    for p_i in points_neighbors:
      ref_i = ref_keys.index(p_i)
      if p_i in exterior_points: # Known Solution
        laplace_beltrami[ref_i, ref_i] = 1
        laplace_vs[ref_i] = p_i.position()
      else: # Unknown We Solve for
        for p_j in points_neighbors[p_i]:
          ref_j = ref_keys.index(p_j)
          if unord_hash(p_i.number(), p_j.number()) in hole_edges_neighbors:
            poly_1, poly_2 = hole_edges_neighbors[unord_hash(p_i.number(), p_j.number())]
            poly_1_p, poly_2_p = None, None
            for poly_e in poly_1.get_edges():
              if p_i in poly_e and not p_j in poly_e:
                poly_1_p = poly_e[0] if poly_e[0] != p_i else poly_e[1]
            for poly_e in poly_2.get_edges():
              if p_i in poly_e and not p_j in poly_e:
                poly_2_p = poly_e[0] if poly_e[0] != p_i else poly_e[1]
            # Treat quadrilateral (i,k,l,j) as two triangles, get angle_ikj not angle_ikl
            e_i1 = poly_1_p.position() - p_i.position()
            e_1j = p_j.position() - poly_1_p.position()
            e_i2 = poly_2_p.position() - p_i.position()
            e_2j = p_j.position() - poly_2_p.position()
            angle_1 = math.radians(e_i1.angleTo(e_1j))
            angle_2 = math.radians(e_i2.angleTo(e_2j))
            cot_angle_1 = math.cos(angle_1) / math.sin(angle_1)
            cot_angle_2 = math.cos(angle_2) / math.sin(angle_2)
            laplace_beltrami[ref_i, ref_j] = (cot_angle_1 + cot_angle_2)
        laplace_beltrami[ref_i, ref_i] = -1 * sum(laplace_beltrami[ref_i])

    laplace_fs = np.zeros((len(points_neighbors), 3))
    for dim in range(laplace_vs.shape[1]):
      sol, is_singular = lss(laplace_beltrami, np.transpose(laplace_vs[:, dim]))
      if is_singular:
        laplace_fs[:, dim] = sol
      
    for p in points_neighbors:
      ref = ref_keys.index(p)
      p.setPosition(laplace_fs[ref])
      if p not in exterior_points:
        p.setPosition(laplace_fs[ref])

class Moving_Least_Squares_Fill():
  def __init__(self, geo, points, points_vicinity, edges, edges_neighbors):
    self.geo = geo
    self.points = points
    self.points_vicinity = points_vicinity
    self.edges = edges
    self.edges_neighbors = edges

  def next_clockwise_p(self, p):
    vs = p.vertices()
    ps_v2, ps_v4 = [], []
    for v in vs:
      prim = v.prim()
      if prim.type() == hou.primType.Polygon:
        prim_vs = list(prim.vertices())
        prim_v_ind = prim_vs.index(v)
        prim_vs = prim_vs[prim_v_ind:] + prim_vs[:prim_v_ind]
        ps_v2.append(prim_vs[1].point())
        ps_v4.append(prim_vs[3].point())
    next_ps = list(set(ps_v2) - set(ps_v4))
    assert (len(next_ps) == 1)
    return next_ps[0]

  def uv_to_xy(self, u, v):
    # Mapping: (u - u_min) * scale, (v - v_min) * scale
    return ((u - self.u_min) * self.scale, (v - self.v_min) * self.scale)
  
  def fill(self):
    if is_debug: start = time.time()
    '''
    We Follow J Wang, M M Oliverira [2006],
    filling holes on locally smooth surfaces reconstructed from point clouds

    A. Compute a reference Plane,
       with origin O and coordinates U, V, S
       where
        O = (x_het, y_het, z_het), average of vicinity points
        U,V,S = (eigenvector with largest eigenvalue of M^TM,
                 eigenvector with middling eigenvalue of M^TM,
                 eigenvector with smallest eigenvalue of M^TM)
        where 
           M = | x1-x_het  y1-y_het  z1-z_het |
               | x2-x_het  y2-y_het  z2-z_het |
               |           ........           |
               | xn-x_het  yn-y_het  zn-z_het |
    '''
    O = hou.Vector3(0, 0, 0)
    P_boundary = []
    coplanar, coplanar_ind = [self.points[0].position()] * (len(self.points)-1), 0
    for point in self.points:
      position = point.position()
      O += position
      P_boundary.append(position)
      if coplanar_ind >= 1:
        coplanar[coplanar_ind-1] = position - coplanar[coplanar_ind-1]
      coplanar_ind += 1

    P_vicinity = []
    for point_vicinity in list(set(self.points_vicinity) - set(self.points)):
      position = point_vicinity.position()
      O += position
      P_vicinity.append(position)
    O = np.array(O / len(points_vicinity))
    P_boundary, P_vicinity = np.array(P_boundary), np.array(P_vicinity)
    P = np.concatenate((P_boundary, P_vicinity))
    M = P - O
    
    scaling = math.ceil(math.log10(len(points)))
    coplanar = np.array(coplanar)
    tol = 1e-15 * math.pow(10, scaling * rank_factor)
    rank = np.linalg.matrix_rank(coplanar, tol=tol)
    # NOTE: default tolerance fails majority of time, chosen tolerance
    # based on limited testing.
    is_coplanar = (rank <= 2)
    if is_coplanar:
      print("hole boundary coplanar, rank: " + str(rank) + " If this is incorrect, decrease rank tolerance factor: " + str(tol), flush=True)
      # If points are coplanar, then s = v1 x v2, u = v1, v = s x u
      s = np.cross(coplanar[0], coplanar[len(coplanar)-1])
      s = s / math.sqrt(math.pow(s[0],2) + math.pow(s[1],2) + math.pow(s[2],2))
      u = coplanar[0]
      u = u / math.sqrt(math.pow(u[0],2) + math.pow(u[1],2) + math.pow(u[2],2))
      v = np.cross(s, u)
    else:
      print("hole boundary not coplanar, rank: " + str(rank) + " If this is incorrect, increase rank tolerance factor: " + str(tol), flush=True)
      # Else, u, v, s can be computed via SVD
      MTM = np.matmul(np.transpose(M), M)
      eigenvalues, eigenvectors = np.linalg.eig(MTM)
      order = eigenvalues.argsort()
      u, v, s = eigenvectors[order[2]], eigenvectors[order[1]], eigenvectors[order[0]]
    '''
    B. Project each vicinity point p orthographically onto reference plane to get p_
       and distance s. Convert 3D p_ point to 2D (u, v) coordinate
    '''
    F_boundary, F_vicinity = np.inner((P_boundary - O), s), np.inner((P_vicinity - O), s)
    F = np.concatenate(((F_boundary, F_vicinity)))
    P_boundary_ = P_boundary - np.multiply(np.tile(s, (F_boundary.shape[0], 1)), F_boundary[:, np.newaxis])
    P_vicinity_ = P_vicinity - np.multiply(np.tile(s, (F_vicinity.shape[0], 1)), F_vicinity[:, np.newaxis])
    P_ = np.concatenate((P_boundary_, P_vicinity_))

    U_boundary, V_boundary = np.inner((P_boundary_ - O), u), np.inner((P_boundary_ - O), v)
    U_vicinity, V_vicinity = np.inner((P_vicinity_ - O), u), np.inner((P_vicinity_ - O), v)
    U, V = np.concatenate((U_boundary, U_vicinity)), np.concatenate((V_boundary, V_vicinity))
    UV = np.vstack((U, V)).T
    '''
    C. Then we generate a mask image. Unlike the original implementation, we only use
       the boundary points.
    D. New sampling points are then set over a regular grid in UV space.
       where
         grid_stepsize = sqrt(area/3n)
         where
           area = sum area of polygons connected to these points
           n = number of points on boundary
       We reject a sampling point if it is closer than beta * grid_stepsize from boundary
       where
         beta = 0.75    
    '''
    # NOTE: We assume area is in 3D, unprojected space
    polys = defaultdict(int)
    for point in points:
      for prim in point.prims():
        if prim.type() == hou.primType.Polygon:
          polys[prim.number()] = prim.intrinsicValue("measuredarea")
    grid_stepsize = math.sqrt(sum(polys.values()) / (3 * len(points)) )
    beta = 0.75
    # We need to convert from UV coord system (where (0, 0) is domain center)
    # To image representable form (where (0, 0) is domain start), and have visible results
    self.u_min, self.u_max = np.min(U), np.max(U)
    self.v_min, self.v_max = np.min(V), np.max(V)
    self.scale = int(math.pow(10, scaling * scale_factor))

    img_x, img_y = self.uv_to_xy(self.u_max, self.v_max)
    img = Image.new("L", (math.ceil(img_x), math.ceil(img_y)), "#000000") # Black Fill = Mask
    draw = ImageDraw.Draw(img)

    pix_boundary = []
    pix_inner_boundary = []
    for u_boundary, v_boundary in zip(U_boundary, V_boundary):
      # NOTE: This is wrong-- Bad approximation
      u_inner, v_inner = np.array([u_boundary, v_boundary]) + (beta * grid_stepsize * np.array([-1*u_boundary, -1*v_boundary])
        * (1 / math.sqrt(math.pow(u_boundary, 2) + math.pow(v_boundary, 2))))
      pix_inner_pos = self.uv_to_xy(u_inner, v_inner)
      pix_inner_boundary.append(pix_inner_pos)

      pix_pos = self.uv_to_xy(u_boundary, v_boundary)
      pix_boundary.append(pix_pos)
    draw.polygon(pix_boundary, fill="#7F7F7F") # Gray Fill = Opening
    draw.point(pix_boundary, fill="#CCCCCC") # Light Gray points = Boundary points
    draw.polygon(pix_inner_boundary, fill="#FFFFFF") # White Fill = Safe

    sample_points = defaultdict(list)
    u_ind = 0
    for u_pos in np.arange(self.u_min, self.u_max, grid_stepsize):
      v_ind = 0
      for v_pos in np.arange(self.v_min, self.v_max, grid_stepsize):
        pix_pos = self.uv_to_xy(u_pos, v_pos)
        if img.getpixel(pix_pos) == 255:
          draw.point(pix_pos, fill="#000000") # Black Points = New Sampling Points
          sample_points[u_ind, v_ind] = (u_pos, v_pos)
        v_ind += 1
      u_ind += 1

    path_name = hou.hipFile.name().split(".")[0]
    img.save(path_name + "/" + "see_new_sampling.png")
    '''
    E. Fit a surface through this height field using Moving Least Squares
       For each new sample point p, (u, v), we will learn a function S:
         S(u, v) = a0 + a1u + a2v + a3u^2 + a4v^2 + a5uv 
       We then use S to compute ideal position of p in 3d space:
         ideal_p = [u', v', S(u, v)*s]
       
       S can be learned via fitness function:
         E(S) = SUMi_1,N wi(p) (S(pi)-fi)^2
         where
            pi = (ui, vi) position of ith point
            fi = projected distance of ith point
            wi(p) = e^-alpha*di(p)^2
                    ---------------
                        di(p)^2
            where 
              di(p) = distance from p to pi in uv space
       In turn, the solution for a0....5 where E(S) is minimized is:
         a(p) = (BW(p)B^T)^-1 BW(p)F
         where
           B = | 1.....1    | W(p) = | w1    0 | F = | f1 |
               | u1....un   |        |   ...   |     | .. |
               | v1....vn   |        | 0    wn |     | fn |
               | u1^2..un^2 |
               | v1^2..vn^2 |  
               | u1v1..unvn |
    '''
    patch_points = []
    B = np.concatenate((np.ones(U.shape[0]), U, V, U*U, V*V, U*V)).reshape((6, U.shape[0]))
    sample_to_proj_point= defaultdict(hou.Point)
    min_uv = (float('inf'), float('inf'))
    for uv_ind, sample_point in sample_points.items():
      u_ind, v_ind = uv_ind
      u_sample, v_sample = sample_point

      D = np.sum(np.power((UV-sample_point), 2), axis=1)
      W = np.diag(np.exp(-1*D)/(D))
      A = np.matmul(np.linalg.inv(np.matmul(B, np.matmul(W, B.T))), np.matmul(B, np.matmul(W, F)))

      # We map from uv space to 3d space, and then project the sample point in direction S
      # by the distance S(u, v)
      s_sample = A[0] + A[1]*u_sample + A[2]*v_sample + A[3]*u_sample*u_sample + A[4]*v_sample*v_sample + A[5]*u_sample*v_sample
      sample_proj_pos = O + u_sample * u + v_sample * v + s_sample * s
      sample_proj_point = geo.createPoint()
      sample_proj_point.setPosition(sample_proj_pos)
      sample_proj_point.setAttribValue("N", hou.Vector3(s).normalized())

      patch_points.append(sample_proj_point)
      sample_to_proj_point[u_ind, v_ind] = sample_proj_point
      if ((u_ind - 1, v_ind - 1) in sample_points and (u_ind - 1, v_ind) in sample_points and (u_ind, v_ind - 1) in sample_points):
        min_uv = min((u_ind - 1, v_ind - 1), min_uv)
        ps = [sample_to_proj_point[u_ind - 1, v_ind], sample_to_proj_point[u_ind, v_ind],
              sample_to_proj_point[u_ind, v_ind - 1], sample_to_proj_point[u_ind - 1, v_ind - 1]]
        '''
        ps = [sample_to_proj_point[u_ind - 1, v_ind - 1], sample_to_proj_point[u_ind, v_ind - 1], 
              sample_to_proj_point[u_ind, v_ind], sample_to_proj_point[u_ind - 1, v_ind]]'''
        poly = get_poly(geo, ps)
    if is_debug: print("Island patch generated in: " + str(time.time() - start), flush=True)
      
    '''
    We now have the original mesh and a new patch mesh. This forms an island hole
    
    We Follow F Bi, Y Hu, X Chen, Y Ma [2013],
    Island hole automatic filling algorithm in triangular meshes
    F. Detect inner boundary points
    '''
    if is_debug: start = time.time()
    start_p = sample_to_proj_point[min_uv]
    inner_ps = [start_p]
    curr_p = self.next_clockwise_p(start_p)

    while curr_p != start_p:
      inner_ps.append(curr_p)
      curr_p = self.next_clockwise_p(curr_p)
    
    '''
    G. Find the two points on the inner boundary furthest from one another
    '''
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
    H. Find each of the points, find the closest point on the outer boundary
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
    G. Connect the inner points with their respective outer point.
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
    H. We can split the two regular holes to more holes, depending on the size of the hole
       Fill the holes using any method
    '''
    max_boundary_size = 200
    boundaries = [(p_1_to_p_2_outer, p_1_to_p_2_inner_rev), (p_2_to_p_1_outer, p_2_to_p_1_inner_rev)] # outer, inner
    num_split = 0
    while boundaries:
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
    if is_debug: print("island hole filled in " + str(time.time() - start) + " with " + str(num_split) + " splits", flush=True)
    return patch_points
    
class AFT_Fill():
  def __init__(self, geo, points, edges, edges_neighbors):
    self.geo = geo
    self.points = points
    self.edges = edges
    self.edges_neighbors = edges_neighbors

  def get_angle(self, p, points_neighbors):
    p_1, p_2 = points_neighbors[p]
    e1, e2 = p_1.position() - p.position(), p_2.position() - p.position()
    sum_angle = 0
    for prim in p.prims():
      if prim.type() == hou.primType.Polygon:
        poly = VirtualPolygon(False, prim)
        poly_edges = []
        for p1, p2 in poly.get_edges():
          if p == p1:
            poly_edges.append(p2.position() - p1.position())
          elif p == p2:
            poly_edges.append(p1.position() - p2.position())
        sum_angle += poly_edges[0].angleTo(poly_edges[1])
      
    if sum_angle < 180:
      return 360 - e1.angleTo(e2)
    else:
      return e1.angleTo(e2)

  def get_Nsectors(self, p, p_1, p_2, n):
    # n:2 = point of bisector, n:3 = points of trisector, etc
    e1, e2 = p_1.position() - p.position(), p_2.position() - p.position()
    len_ave = 0.5 * (e1.length() + e2.length())
    e1, e2 = e1.normalized(), e2.normalized()
    # no sector, direct fill
    if n == 1:
      new_poly = get_poly(self.geo, [p, p_1, p_2])
      return ([], [new_poly])
    # bisector computed using angle bisector theorem
    elif n == 2:
      new_point = self.geo.createPoint()
      eo_prev = len_ave * (e2.length() * e1 + e1.length() * e2).normalized()
      new_point.setPosition(p.position() + eo_prev)

      new_poly_1 = get_poly(self.geo, [p, p_1, new_point])
      new_poly_2 = get_poly(self.geo, [p, new_point, p_2])
        
      get_normal(self.geo, new_point)
      return ([new_point], [new_poly_1, new_poly_2])
    # trisector approximated via successive bisections, 1/3 = 1/4 + 1/16 + 1/64 + ...
    elif n == 3:
      new_point_1, new_point_2 = self.geo.createPoint(), self.geo.createPoint()
      max_steps, epsilon = 10, 0.3
      curr_e1, curr_e2 = e1, e2
      curr_trisector = curr_e1
      for i in range(max_steps * 2):
        prev_trisector = curr_trisector
        curr_trisector = curr_e2.length() * curr_e1 + curr_e1.length() * curr_e2
        if prev_trisector.angleTo(curr_trisector) < epsilon:
          break
        if i % 2 == 0:
          curr_e2 = curr_trisector
        else:
          curr_e1 = curr_trisector
      trisector_1 = curr_trisector
      trisector_2 = (e2.length() * trisector_1 + trisector_1.length() * e2).normalized()
      eo_prev_1, eo_prev_2 = len_ave * trisector_1.normalized(), len_ave * trisector_2

      new_point_1.setPosition(p.position() + eo_prev_1)
      new_point_2.setPosition(p.position() + eo_prev_2)

      new_poly_1 = get_poly(self.geo, [p, p_1, new_point_1])
      new_poly_2 = get_poly(self.geo, [p, new_point_1, new_point_2])
      new_poly_3 = get_poly(self.geo, [p, new_point_2, p_2])

      get_normal(self.geo, new_point_1)
      get_normal(self.geo, new_point_2)
      return ([new_point_1, new_point_2], [new_poly_1, new_poly_2, new_poly_3])

  def correct_normal(self, p, p_1, p_2, points_neighbors):
    n = int((math.ceil(len(points_neighbors)/ float(10))))
    e_dir = [np.zeros(3), np.zeros(3)]
    e_len = [0, 0]

    p_currs = [p_1, p_2]
    for direction in range(2):
      p_prev = p
      p_curr = p_currs[direction]
      for _ in range(n):
        e_curr = p_curr.position() - p.position()
        e_dir[direction] += e_curr
        e_len[direction] += e_curr.length()

        p_a, p_b = points_neighbors[p_curr]
        p_next = p_b if p_a == p_prev else p_a
        p_prev = p_curr
        p_curr = p_next
    e_dir1 = hou.Vector3(e_dir[0] / e_len[0])
    e_dir2 = hou.Vector3(e_dir[1] / e_len[1])

    normal_i = hou.Vector3(p.attribValue("N"))
    # NOTE: Paper says e1 x e2 / ||e1 x e2||, but e2 x e1 / ||e2 x e1|| makes sense
    normal_e = e_dir2.cross(e_dir1).normalized()
    normal_c = (alpha * normal_i + beta * normal_e).normalized()
    return normal_c

  def optimize_new_point(self, p, p_1, p_2, new_points, normal_c):
    # 1. Compute the Taubin Curvature
    # NOTE: ALL_N,E elem R^3, N^T * E == N.E, so we use RHS intead
    e1, e2 = p_1.position() - p.position(), p_2.position() - p.position()
    taubin_curvature = (normal_c.dot(e1) / math.pow(e1.length(), 2)
                      + normal_c.dot(e2) / math.pow(e2.length(), 2))
    # 2. Solve for eo_new, through minimizing F(eo_new)
    for new_point in new_points:
      eo_prev = new_point.position() - p.position()
      def weighted_function(eo_new):
        eo_new = hou.Vector3(eo_new)
        res = (w1 * math.pow((2 * normal_c.dot(eo_new)) / math.pow(eo_new.length(), 2) - taubin_curvature, 2)
              + w2 * math.pow((eo_new - eo_prev).length(), 2))
        return res
      eo_new = hou.Vector3(minimize(weighted_function,  np.array(eo_prev))[0])
      # 3. Calculate optimal new_point
      new_point.setPosition(p.position() + eo_new)
    # 4. Recalculate changed vertex normals
    for new_point in new_points:
      get_normal(self.geo, new_point)


  def lhs_merge(self, p, new_point, point_rhs, p_1, p_2, points_angle, points_neighbors):
    # bisection: new_point = new_point, point_rhs = p_2
    # trisection: new_point = new_point_1, point_rhs = new_point_2
    p_a, _ = get_clockwise_neighbors(new_point, points_neighbors[new_point])
    points_neighbors[p_1].remove(p)
    points_neighbors[p_2].remove(p)
    get_poly(self.geo, [new_point, p, p_1])
    get_poly(self.geo, [new_point, point_rhs, p])

    points_neighbors[new_point] = [p_a, point_rhs]
    if point_rhs == p_2:
      points_neighbors[p_2].append(new_point)
    else:
      points_neighbors[point_rhs] = [new_point, p_2]
      points_neighbors[p_2].append(point_rhs)
      points_angle[point_rhs] = self.get_angle(point_rhs, points_neighbors)
    points_angle[new_point] = self.get_angle(new_point, points_neighbors)
    points_angle[p_2] = self.get_angle(p_2, points_neighbors)
    del points_neighbors[p_1]
    del points_angle[p_1]

  def rhs_merge(self, p, new_point, point_lhs, p_1, p_2, points_angle, points_neighbors):
    # bisection: new_point = new_point, points_lhs = p_1
    # trisection: new_point = new_point_2, point_lhs = new_point_1
    _, p_b = get_clockwise_neighbors(new_point, points_neighbors[new_point])
    points_neighbors[p_1].remove(p)
    points_neighbors[p_2].remove(p)
    get_poly(self.geo, [new_point, p, point_lhs])
    get_poly(self.geo, [new_point, p_2, p])

    points_neighbors[new_point] = [point_lhs, p_b]
    if point_lhs == p_1:
      points_neighbors[p_1].append(new_point)
    else:
      points_neighbors[point_lhs] = [p_1, new_point]
      points_neighbors[p_1].append(point_lhs)
      points_angle[point_lhs] = self.get_angle(point_lhs, points_neighbors)
    points_angle[new_point] = self.get_angle(new_point, points_neighbors)
    points_angle[p_1] = self.get_angle(p_1, points_neighbors)
    del points_neighbors[p_2]
    del points_angle[p_2]

  def center_merge(self, p, new_point, point_other_l, point_other_r, p_1, p_2, points_angle, points_neighbors):
    # Two sub-holes. Compute AFT individually
    # bisection: new_point = new_point, point_other_l = p_1, point_other_r = p_2
    # trisection_left: new_point = new_point_1, point_other_l = p_1, point_other_r = new_point_2
    # trisection_right: new_point = new_point_2, point_other_l = new_point_1, point_other_r = p_2
    p_a, p_b = get_clockwise_neighbors(new_point, points_neighbors[new_point])
    points_neighbors[p_1].remove(p)
    points_neighbors[p_2].remove(p)
    get_poly(self.geo, [new_point, p, point_other_l])
    get_poly(self.geo, [new_point, point_other_r, p])

    p_loops = [(point_other_l, p_b), (p_a, point_other_r)]
    points_neighbors[p_1].append(new_point if point_other_l == p_1 else point_other_l)
    points_neighbors[p_2].append(new_point if point_other_r == p_2 else point_other_r)
    if point_other_l == p_1 and point_other_r != p_2:
      points_neighbors[point_other_r] = [new_point, p_2]
      points_angle[point_other_r] = self.get_angle(point_other_r, points_neighbors)
    elif point_other_l != p_1 and point_other_r == p_2:
      points_neighbors[point_other_l] = [p_1, new_point]
      points_angle[point_other_l] = self.get_angle(point_other_l, points_neighbors)
    points_angle[p_1] = self.get_angle(p_1, points_neighbors)
    points_angle[p_2] = self.get_angle(p_2, points_neighbors)

    points_loops_new, angle_loops_new = [], []
    for p_loop in p_loops:
      points_neighbors_loop = defaultdict(list)
      points_angle_loop = defaultdict(list) 
      p_prev = new_point
      p_curr = p_loop[0]
      while p_curr != new_point:
        p__a, p__b = points_neighbors[p_curr]
        points_neighbors_loop[p_curr] = [p__a, p__b]
        points_angle_loop[p_curr] = points_angle[p_curr]
        p_next = p__b if p__a == p_prev else p__a
        p_prev = p_curr
        p_curr = p_next
      points_neighbors_loop[new_point] = [p_loop[0], p_loop[1]]
      points_angle_loop[new_point] = self.get_angle(new_point, points_neighbors_loop)
      points_loops_new.append(points_neighbors_loop)
      angle_loops_new.append(points_angle_loop)
    return points_loops_new, angle_loops_new

  def fill(self):
    points_neighbors = defaultdict(list)
    for edge in self.edges:
      p_1, p_2 = edge.points()
      points_neighbors[p_1].append(p_2)
      points_neighbors[p_2].append(p_1)
    points_angle = defaultdict(list)
    for p in points_neighbors:
      points_angle[p] = self.get_angle(p, points_neighbors)

    emergency_stop = False
    marked_for_deletion = []
    points_loops, angle_loops = [points_neighbors], [points_angle]

    patch_points = []
    while len(points_loops) > 0 and not emergency_stop:
      points_neighbors, points_angle = points_loops[0], angle_loops[0]
      i = 0
      while len(points_neighbors) >= 3:
        #print("iter: " + str(i) + " remaining: " + str(len(points_neighbors)))
        if is_iter_threshold and i > iter_threshold:
          emergency_stop = True
          break
        p = min(points_angle, key=points_angle.get)
        min_angle = points_angle[p]
        '''
        A. Weighted Roulette
          Choose any point with angle < angle_threshold. Lets more borders contribute to AFT
          Disable to remove randomness
        '''
        if min_angle <= angle_threshold and is_angle_threshold:
          p_roulette = []
          for p_ in points_angle:
            point_angle = points_angle[p_]
            if point_angle <= angle_threshold:
              p_roulette += [p_] * int(angle_threshold - point_angle + 1)
          p = random.choice(p_roulette)
          min_angle = points_angle[p]
        
        p_1, p_2 = get_clockwise_neighbors(p, points_neighbors[p])
        '''
        B. Correct normal
        '''
        normal_c = self.correct_normal(p, p_1, p_2, points_neighbors)
        p.setAttribValue("N", normal_c)
        '''
        C. Basic AFT
          angle <= 75:       fill with no points, single triangle
          75 < angle <= 135: fill with one point, two triangles, along bisector
          otherwise:         fill with two points, three triangles, along trisector

        D. Optimize new points
        E. Point Distance Thresholding
          merge points along boundary that are within point_threshold of one another
        '''
        if min_angle <= 75:
          points_neighbors[p_1].remove(p)
          points_neighbors[p_2].remove(p)
          self.get_Nsectors(p, p_1, p_2, 1)
          self.optimize_new_point(p, p_1, p_2, [], normal_c)
          points_neighbors[p_1].append(p_2)
          points_neighbors[p_2].append(p_1)
          points_angle[p_1] = self.get_angle(p_1, points_neighbors)
          points_angle[p_2] = self.get_angle(p_2, points_neighbors)
        elif min_angle <= 135:
          new_points, new_polys = self.get_Nsectors(p, p_1, p_2, 2)
          new_point = new_points[0]
          self.optimize_new_point(p, p_1, p_2, [new_point], normal_c)
          is_merged = False
          if is_point_threshold:
            p_threshold = point_threshold * (new_point.position() - p.position()).length()
            for p_ in points_neighbors:
              distance = (p_.position() - new_point.position()).length()
              if distance < p_threshold and not p_ in [p, p_1, p_2]:
                marked_for_deletion.extend(new_points)
                self.geo.deletePrims(new_polys, keep_points=True)
                new_point, is_merged = p_, True
                break 
          if is_merged:
            p_a, p_b = get_clockwise_neighbors(new_point, points_neighbors[new_point])
            if p_1 == p_b:
              self.lhs_merge(p, new_point, p_2, p_1, p_2, points_angle, points_neighbors)
            elif p_2 == p_a:
              self.rhs_merge(p, new_point, p_1, p_1, p_2, points_angle, points_neighbors)
            else:
              points_loops_new, angle_loops_new = self.center_merge(p, new_point, p_1, p_2, p_1, p_2, points_angle, points_neighbors)
              points_loops.extend(points_loops_new)
              angle_loops.extend(angle_loops_new)
              break
          else:
            points_neighbors[p_1].remove(p)
            points_neighbors[p_2].remove(p)
            points_neighbors[new_point] = [p_1, p_2]
            points_neighbors[p_1].append(new_point)
            points_neighbors[p_2].append(new_point)
            points_angle[new_point] = self.get_angle(new_point, points_neighbors)
            points_angle[p_1] = self.get_angle(p_1, points_neighbors)
            points_angle[p_2] = self.get_angle(p_2, points_neighbors)
            patch_points.append(new_point)
        else:
          new_points, new_polys = self.get_Nsectors(p, p_1, p_2, 3)
          new_point_1, new_point_2 = new_points
          self.optimize_new_point(p, p_1, p_2, [new_point_1, new_point_2], normal_c)
          is_merged_1, is_merged_2 = False, False
          if is_point_threshold:
            p_threshold_1 = point_threshold * (new_point_1.position() - p.position()).length()
            p_threshold_2 = point_threshold * (new_point_2.position() - p.position()).length()
            new_point_1_, new_point_2_ = None, None
            points_to_delete, polys_to_delete = [], []
            for p_ in points_neighbors:
              distance_1 = (p_.position() - new_point_1.position()).length() if not is_merged_1 else distance_1
              distance_2 = (p_.position() - new_point_2.position()).length() if not is_merged_2 else distance_2
              if distance_1 < p_threshold_1 and not p_ in [p, p_1, p_2, new_point_2] and not is_merged_1:
                points_to_delete.append(new_point_1)
                new_point_1_, is_merged_1 = p_, True
              if distance_2 < p_threshold_2 and not p_ in [p, p_1, p_2, new_point_1] and not is_merged_2:
                points_to_delete.append(new_point_2)
                new_point_2_, is_merged_2 = p_, True
              if is_merged_1 and is_merged_2:
                if distance_1 < distance_2:
                  new_point_2_, is_merged_2 = new_point_2, False
                  points_to_delete.remove(new_point_2)
                else:
                  new_point_1_, is_merged_1 = new_point_1, False
                  points_to_delete.remove(new_point_1)
                
            if new_point_1 not in points_to_delete:
              patch_points.append(new_point_1)
            if new_point_2 not in points_to_delete:
              patch_points.append(new_point_2)

            for new_poly in new_polys:
              for point_to_delete in points_to_delete:
                if point_to_delete in new_poly.points() and new_poly not in polys_to_delete:
                  polys_to_delete.append(new_poly)
            polys_to_delete = np.array(polys_to_delete)
            marked_for_deletion.extend(points_to_delete)
            self.geo.deletePrims(polys_to_delete, keep_points=True)

            new_point_1 = new_point_1_ if new_point_1_ != None else new_point_1
            new_point_2 = new_point_2_ if new_point_2_ != None else new_point_2
          if is_merged_1:
            p_a, p_b = get_clockwise_neighbors(new_point_1, points_neighbors[new_point_1])
            if p_1 == p_b:
              self.lhs_merge(p, new_point_1, new_point_2, p_1, p_2, points_angle, points_neighbors)
            else:
              points_loops_new, angle_loops_new = self.center_merge(p, new_point_1, p_1, new_point_2, p_1, p_2, points_angle, points_neighbors)
              points_loops.extend(points_loops_new)
              angle_loops.extend(angle_loops_new)
              break
          elif is_merged_2:
            p_a, p_b = get_clockwise_neighbors(new_point_2, points_neighbors[new_point_2])
            if p_2 == p_a:
              self.rhs_merge(p, new_point_2, new_point_1, p_1, p_2, points_angle, points_neighbors)  
            else:
              points_loops_new, angle_loops_new = self.center_merge(p, new_point_2, new_point_1, p_2, p_1, p_2, points_angle, points_neighbors)
              points_loops.extend(points_loops_new)
              angle_loops.extend(angle_loops_new)
              break
          else:
            points_neighbors[p_1].remove(p)
            points_neighbors[p_2].remove(p)      
            points_neighbors[p_1].append(new_point_1)
            points_neighbors[p_2].append(new_point_2)
            points_neighbors[new_point_1] = [p_1, new_point_2]
            points_neighbors[new_point_2] = [new_point_1, p_2]
            points_angle[new_point_1] = self.get_angle(new_point_1, points_neighbors)
            points_angle[new_point_2] = self.get_angle(new_point_2, points_neighbors)
            points_angle[p_1] = self.get_angle(p_1, points_neighbors)
            points_angle[p_2] = self.get_angle(p_2, points_neighbors)
        del points_angle[p]
        del points_neighbors[p]
        i += 1
      if len(points_neighbors) == 3:
        get_poly(self.geo, points_neighbors.keys())
      points_loops.remove(points_neighbors)
      angle_loops.remove(points_angle)

    self.geo.deletePoints(marked_for_deletion)
    return patch_points

# ------------ Main Code ------------ #
# - Fill Type Parameters
small = hou.session.find_parm(hou.parent(), "low_small_type")
med = hou.session.find_parm(hou.parent(), "low_med_type")
large = hou.session.find_parm(hou.parent(), "low_large_type")

# - Improved AFT Parameters
alpha = hou.session.find_parm(hou.parent(), "low_alpha_beta")
beta = 1 - alpha
w1 = hou.session.find_parm(hou.parent(), "low_w1_w2")
w2 = 1 - w1
is_iter_threshold = bool(hou.session.find_parm(hou.parent(), "isIter"))
iter_threshold = hou.session.find_parm(hou.parent(), "low_iter_threshold")
is_angle_threshold = bool(hou.session.find_parm(hou.parent(), "isAngle"))
angle_threshold = hou.session.find_parm(hou.parent(), "low_angle_threshold")
is_point_threshold = bool(hou.session.find_parm(hou.parent(), "isDistance"))
point_threshold = hou.session.find_parm(hou.parent(), "low_distance_threshold")

# - MLS with MWT Parameters
rank_factor = hou.session.find_parm(hou.parent(), "low_rank_factor")
scale_factor = hou.session.find_parm(hou.parent(), "low_scale_factor")

# NOTE: points ordered, but ordering breaks after deletion.
#       Min triangulation relies on ordering
saved_groups = []
for i in range(1, len(point_boundaries)):
  points = point_boundaries[i].points()
  points_vicinity = neighbor_point_boundaries[i].points()
  edges = edge_boundaries[i].edges()
  edges_neighbors = neighbor_edge_boundaries[i].edges()

  if len(points) <= 8:
    '''
    1. Fill small holes with centroid-based method
    '''
    if small == 0:
      MinTriangulation(geo, points).min_triangulation(generate=True)
    else:
      Centroid_Fill(geo, points, edges).fill()
  elif len(points) <= 20:
    '''
    2. Fill Medium hole with projection-based method
    '''
    if med == 0:
      Projection_BiLaplacian_Fill(geo, points, edges, edges_neighbors).fill()
  else:
    '''
    3. Fill large hole with advancing front method
    '''
    saved_groups.append(point_boundaries[i].name())
    patch_points = geo.createPointGroup("patch_" + point_boundaries[i].name())
    if large == 0:
      patch_points.add(Moving_Least_Squares_Fill(geo, points, points_vicinity, edges, edges_neighbors).fill())
    else:
      patch_points.add(AFT_Fill(geo, points, edges, edges_neighbors).fill())
print(saved_groups)
node.bypass(True)