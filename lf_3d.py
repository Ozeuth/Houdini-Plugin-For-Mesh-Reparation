import hou
import itertools
import math
import numpy as np
import operator
import random
import copy
from collections import defaultdict

node = hou.pwd()
geo = node.geometry()
boundaries = geo.pointGroups()
edge_boundaries = []
neighbor_boundaries = []
for edge_group in geo.edgeGroups():
  if "neighbor" in edge_group.name():
    neighbor_boundaries.append(edge_group)
  else:
    edge_boundaries.append(edge_group)

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

# NOTE: Houdini 2020 does not support scipy, using FC's np-only minimizer
def minimize(f, x_start,
            step=0.1, no_improve_thr=10e-6,
            no_improv_break=10, max_iter=0,
            alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
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

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres

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
    for p_zip1, p_zip2 in np.array(zip(ps_zip1, ps_zip2)):
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

# NOTE: points ordered, but ordering breaks after deletion.
#       Min triangulation relies on ordering
for i in range(1, len(boundaries)):
  points = boundaries[i].points()
  edges = edge_boundaries[i].edges()
  edges_neighbors = neighbor_boundaries[i].edges()
  '''
  2. Fill small holes with centroid-based method
                       | 1 if h has <= 6 points in it
    is_small_hole(h) = | 0 otherwise
  '''
  # TODO: Change small hole definition to n-gon?
  if len(points) <= 8:
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
      ps = list(edge.points()) + [centroid]
      get_poly(geo, ps)
  elif len(points) <= 14:
    '''
    3. Fill Medium hole with projection-based method
    3A. Initialize with minimum area triangulation
    '''
    e_lens_hashed = {}
    for i in range(len(points)):
      for j in range(i+1, len(points)):
        p_i, p_j = points[i], points[j]
        pi_pos = p_i.position()
        pj_pos = p_j.position()
        e_lens_hashed[unord_hash(p_i.number(), p_j.number())] = (pi_pos - pj_pos).length()

    class MinTriangulation():
      def __init__(self, geo, points, cache_costs=None):
        if cache_costs is None:
          cache_costs = {}
          for i in range(len(points)):
            for j in range(i+1, len(points)):
              p_i, p_j = points[i], points[j]
              pi_pos = p_i.position()
              pj_pos = p_j.position()
              cache_costs[unord_hash(p_i.number(), p_j.number())] = (pi_pos - pj_pos).length()
        self.geo = geo
        self.points = points
        self.cache_costs = cache_costs

      def tri_cost(self, i, j, k, is_mwt=True):
        eik_len = self.cache_costs[unord_hash(points[i].number(), points[k].number())]
        ekj_len = self.cache_costs[unord_hash(points[k].number(), points[j].number())]
        if is_mwt:
          return eik_len + ekj_len
        else:
          eij_len = self.cache_costs[unord_hash(i, j)]
          s = eij_len + eik_len + ekj_len / 2
          return math.sqrt(s*(s-eij_len)*(s-eik_len)*(s-ekj_len))

      def tri_min(self, i, j):
        if j <= i+1:
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
        return min_cost, min_polys

      def min_triangulation(self, generate=True):
        _, min_polys = self.tri_min(0, len(self.points)-1)
        if generate:
          for min_poly in min_polys:
            new_poly = self.geo.createPolygon()
            for p in min_poly.ps:
              new_poly.addVertex(p)
        return min_polys
    
    min_polys = MinTriangulation(geo, points, cache_costs=e_lens_hashed).min_triangulation(generate=False)
    '''
    3B. Conduct Triangle Splitting
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
    for edges_neighbor in list(set(edges_neighbors) - set(edges)):
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
    for edge in edges:
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
          p_c = geo.createPoint()
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
    3C. Conduct Edge-Swapping
      For two new polygons (p_i, p_j, p_k), (p_i, p_j, p_m) adjacent to interior edge eij,
                                | 1 p_k lies outside circumsphere created with p_i, p_j, p_m
                                |   OR p_m lies outside cirumsphere created with p_i, p_j, p_k
      is_locally_delaunay(ts) = | 0 otherwise
      if two polygons are not locally delaunay, swap edges. Replace eij with ekm.
    '''
    def find_circle(pA, pB, pC):
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

    marked_for_update = {}
    marked_for_deletion = []
    for interior_edge in interior_edges_neighbors:
      poly_1, poly_2 = interior_edges_neighbors[interior_edge]
      common_edge = poly_1.get_common_edge(poly_2)
      poly_1_p = list(set(poly_1.ps) - set(common_edge))[0]
      poly_2_p = list(set(poly_2.ps) - set(common_edge))[0]

      poly_1_circumsphere_c, poly_1_circumsphere_r = find_circle(common_edge[0], poly_1_p, common_edge[1])

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
      get_poly(geo, min_poly.ps)
    '''
    3D. Conduct Patch Fairing.
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
    for edge in edges:
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
  else:
    '''
    4. Fill large hole with advancing front method
    '''
    def clockwise_neighbors(p, points_neighbors):
      # p_1 = left of p, p_2 = right of p
      p_a, p_b = points_neighbors[p]
      for prim in p.prims():
        if prim.type() == hou.primType.Polygon:
          ps = []
          for v in prim.vertices():
            ps.append(v.point())

          curr_p = p_a if p_a in ps else (p_b if p_b in ps else None)
          if curr_p != None:
            p_i = ps.index(p)
            ps =  ps[p_i:] + ps[:p_i]
            assert(ps[0] == p)
            if ps[1] == curr_p:
              p_2 = curr_p
            elif ps[len(ps) - 1] == curr_p:
              p_1 = curr_p
      return p_1, p_2

    def get_angle(p, points_neighbors):
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
        
    def get_Nsectors(p, p_1, p_2, n):
      # n:2 = point of bisector, n:3 = points of trisector, etc
      e1, e2 = p_1.position() - p.position(), p_2.position() - p.position()
      len_ave = 0.5 * (e1.length() + e2.length())
      e1, e2 = e1.normalized(), e2.normalized()
      # no sector, direct fill
      if n == 1:
        new_poly = get_poly(geo, [p, p_1, p_2])
        return ([], [new_poly])
      # bisector computed using angle bisector theorem
      elif n == 2:
        new_point = geo.createPoint()
        eo_prev = len_ave * (e2.length() * e1 + e1.length() * e2).normalized()
        new_point.setPosition(p.position() + eo_prev)

        new_poly_1 = get_poly(geo, [p, p_1, new_point])
        new_poly_2 = get_poly(geo, [p, new_point, p_2])
        
        get_normal(geo, new_point)
        return ([new_point], [new_poly_1, new_poly_2])
      # trisector approximated via successive bisections, 1/3 = 1/4 + 1/16 + 1/64 + ...
      elif n == 3:
        new_point_1, new_point_2 = geo.createPoint(), geo.createPoint()
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

        new_poly_1 = get_poly(geo, [p, p_1, new_point_1])
        new_poly_2 = get_poly(geo, [p, new_point_1, new_point_2])
        new_poly_3 = get_poly(geo, [p, new_point_2, p_2])

        get_normal(geo, new_point_1)
        get_normal(geo, new_point_2)
        return ([new_point_1, new_point_2], [new_poly_1, new_poly_2, new_poly_3])
      
    def correct_normal(p, p_1, p_2, points_neighbors):
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

    def optimize_new_point(p, p_1, p_2, new_points, normal_c):
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
        get_normal(geo, new_point)

    def lhs_merge(p, new_point, point_rhs, p_1, p_2, points_angle, points_neighbors):
      # bisection: new_point = new_point, point_rhs = p_2
      # trisection: new_point = new_point_1, point_rhs = new_point_2
      p_a, _ = clockwise_neighbors(new_point, points_neighbors)
      points_neighbors[p_1].remove(p)
      points_neighbors[p_2].remove(p)
      get_poly(geo, [new_point, p, p_1])
      get_poly(geo, [new_point, point_rhs, p])

      points_neighbors[new_point] = [p_a, point_rhs]
      if point_rhs == p_2:
        points_neighbors[p_2].append(new_point)
      else:
        points_neighbors[point_rhs] = [new_point, p_2]
        points_neighbors[p_2].append(point_rhs)
        points_angle[point_rhs] = get_angle(point_rhs, points_neighbors)
      points_angle[new_point] = get_angle(new_point, points_neighbors)
      points_angle[p_2] = get_angle(p_2, points_neighbors)
      del points_neighbors[p_1]
      del points_angle[p_1]

    def rhs_merge(p, new_point, point_lhs, p_1, p_2, points_angle, points_neighbors):
      # bisection: new_point = new_point, points_lhs = p_1
      # trisection: new_point = new_point_2, point_lhs = new_point_1
      _, p_b = clockwise_neighbors(new_point, points_neighbors)
      points_neighbors[p_1].remove(p)
      points_neighbors[p_2].remove(p)
      get_poly(geo, [new_point, p, point_lhs])
      get_poly(geo, [new_point, p_2, p])

      points_neighbors[new_point] = [point_lhs, p_b]
      if point_lhs == p_1:
        points_neighbors[p_1].append(new_point)
      else:
        points_neighbors[point_lhs] = [p_1, new_point]
        points_neighbors[p_1].append(point_lhs)
        points_angle[point_lhs] = get_angle(point_lhs, points_neighbors)
      points_angle[new_point] = get_angle(new_point, points_neighbors)
      points_angle[p_1] = get_angle(p_1, points_neighbors)
      del points_neighbors[p_2]
      del points_angle[p_2]

    def center_merge(p, new_point, point_other_l, point_other_r, p_1, p_2, points_angle, points_neighbors):
      # Two sub-holes. Compute AFT individually
      # bisection: new_point = new_point, point_other_l = p_1, point_other_r = p_2
      # trisection_left: new_point = new_point_1, point_other_l = p_1, point_other_r = new_point_2
      # trisection_right: new_point = new_point_2, point_other_l = new_point_1, point_other_r = p_2
      p_a, p_b = clockwise_neighbors(new_point, points_neighbors)
      points_neighbors[p_1].remove(p)
      points_neighbors[p_2].remove(p)
      get_poly(geo, [new_point, p, point_other_l])
      get_poly(geo, [new_point, point_other_r, p])

      p_loops = [(point_other_l, p_b), (p_a, point_other_r)]
      points_neighbors[p_1].append(new_point if point_other_l == p_1 else point_other_l)
      points_neighbors[p_2].append(new_point if point_other_r == p_2 else point_other_r)
      if point_other_l == p_1 and point_other_r != p_2:
        points_neighbors[point_other_r] = [new_point, p_2]
        points_angle[point_other_r] = get_angle(point_other_r, points_neighbors)
      elif point_other_l != p_1 and point_other_r == p_2:
        points_neighbors[point_other_l] = [p_1, new_point]
        points_angle[point_other_l] = get_angle(point_other_l, points_neighbors)
      points_angle[p_1] = get_angle(p_1, points_neighbors)
      points_angle[p_2] = get_angle(p_2, points_neighbors)

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
        points_angle_loop[new_point] = get_angle(new_point, points_neighbors_loop)
        points_loops_new.append(points_neighbors_loop)
        angle_loops_new.append(points_angle_loop)
      return points_loops_new, angle_loops_new

    points_neighbors = defaultdict(list)
    for edge in edges:
      p_1, p_2 = edge.points()
      points_neighbors[p_1].append(p_2)
      points_neighbors[p_2].append(p_1)
    points_angle = defaultdict(list)
    for p in points_neighbors:
      points_angle[p] = get_angle(p, points_neighbors)

    emergency_stop = False
    marked_for_deletion = []
    points_loops, angle_loops = [points_neighbors], [points_angle]
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
        4A. Weighted Roulette
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
        
        p_1, p_2 = clockwise_neighbors(p, points_neighbors)
        '''
        4B. Correct normal
        '''
        normal_c = correct_normal(p, p_1, p_2, points_neighbors)
        p.setAttribValue("N", normal_c)
        '''
        4C. Basic AFT
          angle <= 75:       fill with no points, single triangle
          75 < angle <= 135: fill with one point, two triangles, along bisector
          otherwise:         fill with two points, three triangles, along trisector

        4D. Optimize new points
        4E. Point Distance Thresholding
          merge points along boundary that are within point_threshold of one another
        '''
        if min_angle <= 75:
          points_neighbors[p_1].remove(p)
          points_neighbors[p_2].remove(p)
          get_Nsectors(p, p_1, p_2, 1)
          optimize_new_point(p, p_1, p_2, [], normal_c)
          points_neighbors[p_1].append(p_2)
          points_neighbors[p_2].append(p_1)
          points_angle[p_1] = get_angle(p_1, points_neighbors)
          points_angle[p_2] = get_angle(p_2, points_neighbors)
        elif min_angle <= 135:
          new_points, new_polys = get_Nsectors(p, p_1, p_2, 2)
          new_point = new_points[0]
          optimize_new_point(p, p_1, p_2, [new_point], normal_c)
          is_merged = False
          if is_point_threshold:
            p_threshold = point_threshold * (new_point.position() - p.position()).length()
            for p_ in points_neighbors:
              distance = (p_.position() - new_point.position()).length()
              if distance < p_threshold and not p_ in [p, p_1, p_2]:
                marked_for_deletion.extend(new_points)
                geo.deletePrims(new_polys, keep_points=True)
                new_point, is_merged = p_, True
                break 
          if is_merged:
            p_a, p_b = clockwise_neighbors(new_point, points_neighbors)
            if p_1 == p_b:
              lhs_merge(p, new_point, p_2, p_1, p_2, points_angle, points_neighbors)
            elif p_2 == p_a:
              rhs_merge(p, new_point, p_1, p_1, p_2, points_angle, points_neighbors)
            else:
              points_loops_new, angle_loops_new = center_merge(p, new_point, p_1, p_2, p_1, p_2, points_angle, points_neighbors)
              points_loops.extend(points_loops_new)
              angle_loops.extend(angle_loops_new)
              break
          else:
            points_neighbors[p_1].remove(p)
            points_neighbors[p_2].remove(p)
            points_neighbors[new_point] = [p_1, p_2]
            points_neighbors[p_1].append(new_point)
            points_neighbors[p_2].append(new_point)
            points_angle[new_point] = get_angle(new_point, points_neighbors)
            points_angle[p_1] = get_angle(p_1, points_neighbors)
            points_angle[p_2] = get_angle(p_2, points_neighbors)
        else:
          new_points, new_polys = get_Nsectors(p, p_1, p_2, 3)
          new_point_1, new_point_2 = new_points
          optimize_new_point(p, p_1, p_2, [new_point_1, new_point_2], normal_c)
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
                break
            for new_poly in new_polys:
              for point_to_delete in points_to_delete:
                if point_to_delete in new_poly.points():
                  polys_to_delete.append(new_poly)
            polys_to_delete = np.unique(np.array(polys_to_delete))
            marked_for_deletion.extend(points_to_delete)
            geo.deletePrims(polys_to_delete, keep_points=True)

            new_point_1 = new_point_1_ if new_point_1_ != None else new_point_1
            new_point_2 = new_point_2_ if new_point_2_ != None else new_point_2
          if is_merged_1:
            p_a, p_b = clockwise_neighbors(new_point_1, points_neighbors) 
            if p_1 == p_b:
              lhs_merge(p, new_point_1, new_point_2, p_1, p_2, points_angle, points_neighbors)
            else:
              points_loops_new, angle_loops_new = center_merge(p, new_point_1, p_1, new_point_2, p_1, p_2, points_angle, points_neighbors)
              points_loops.extend(points_loops_new)
              angle_loops.extend(angle_loops_new)
              break
          elif is_merged_2:
            p_a, p_b = clockwise_neighbors(new_point_2, points_neighbors)
            if p_2 == p_a:
              rhs_merge(p, new_point_2, new_point_1, p_1, p_2, points_angle, points_neighbors)  
            else:
              points_loops_new, angle_loops_new = center_merge(p, new_point_2, new_point_1, p_2, p_1, p_2, points_angle, points_neighbors)
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
            points_angle[new_point_1] = get_angle(new_point_1, points_neighbors)
            points_angle[new_point_2] = get_angle(new_point_2, points_neighbors)
            points_angle[p_1] = get_angle(p_1, points_neighbors)
            points_angle[p_2] = get_angle(p_2, points_neighbors)
        del points_angle[p]
        del points_neighbors[p]
        i += 1
      if len(points_neighbors) == 3:
        get_poly(geo, points_neighbors.keys())
      points_loops.remove(points_neighbors)
      angle_loops.remove(points_angle)

    geo.deletePoints(marked_for_deletion)

node.bypass(True)