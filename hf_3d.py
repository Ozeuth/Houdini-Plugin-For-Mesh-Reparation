import hou
import math
import numpy as np
from hausdorff import hausdorff_distance
from collections import defaultdict
from queue import PriorityQueue

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

class Pair():
  def __init__(self, point, elem, inter):
    self.point = point
    self.elem = elem
    self.inter = inter

  def __gt__(self, other):
    return self.point.number() > other.point.number()
  
  def __eq__(self, other):
    return self.point.number() == other.point.number()

  def __repr__(self):
    return (repr((point, elem)))

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

class GapContraction():
  def __init__(self, geo, points, edges=None, is_debug=False):
    self.geo = geo
    self.points = points
    # NOTE: edges must be defined if points not ordered
    self.edges = edges
    self.is_debug = is_debug
  
  def sort_points(self, points):
    points_copy = list(points)
    points_copy.sort(key=lambda x: x.number())
    return tuple(points_copy)

  def min_dist_and_elem(self, point, neighbors_edges_pairs, epsilon=0.1):
    min_dist, min_elem, min_inter = float('inf'), None, None
    virtual_edges = []
    for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
      if point in points_neighbor_:
        virtual_edges += virtual_edges_

    for virtual_edge in virtual_edges:
      '''
            p1
            | proj
      p_inter|_____pi
            |
            p2
      '''
      p1, p2 = virtual_edge
      if p1 != point and p2 != point:
        e_1i = point.position() - p1.position()
        e_12 = p2.position() - p1.position()
        e_1inter = (e_1i.dot(e_12) / e_12.dot(e_12)) * e_12
        mu = 0
        for i in range(3):
          mu += e_1inter[i] / e_12[i] if e_12[i] != 0 else 0
        mu /= 3
        is_self_merge = False
        for prim in point.prims():
          if prim.type() == hou.primType.Polygon:
            if p1 in prim.points() and p2 in prim.points():
              is_self_merge = True
        if epsilon < mu and mu < (1 - epsilon) and not is_self_merge: # Viable Edge
          proj = e_1i - e_1inter
          proj_dist = proj.length()
          p_inter = p1.position() + e_1inter
          if proj_dist < min_dist:
            min_dist, min_elem, min_inter = proj_dist, virtual_edge, p_inter
        else: # Viable Point
          p_1i_dist, p_2i_dist = point.position().distanceTo(p1.position()), point.position().distanceTo(p2.position())
          if p_1i_dist < min_dist:
            min_dist, min_elem = p_1i_dist, p1
          if p_2i_dist < min_dist:
            min_dist, min_elem = p_2i_dist, p2
    return (min_dist, min_elem, min_inter)

  def fill(self):
    '''
    We Follow P Borodin, M Novotni, R Klein [2002],
    Progressive Gap Closing for Mesh Repairing
    '''
    dist_to_pairs = PriorityQueue()
    points_to_elem, elems_to_points = {}, defaultdict(list)
    points_neighbors, virtual_edges = defaultdict(list), []

    neighbors_pull_weight = 1
    neighbors_pull, neighbors_pull_count = defaultdict(list), defaultdict(int)

    if not self.edges:
      for i, point in enumerate(self.points):
        p1 = self.points[i-1 if i > 0 else len(self.points)-1]
        p2 = self.points[i+1 if i < len(self.points)-1 else 0]
        points_neighbors[point] = list(get_clockwise_neighbors(point, (p1, p2)))
        virtual_edges.append(self.sort_points((point, points_neighbors[point][0])))
    else:
      for edge in self.edges:
        p1, p2 = edge.points()
        points_neighbors[p1].append(p2)
        points_neighbors[p2].append(p1)
        if len(points_neighbors[p1]) == 2:
          points_neighbors[p1] = list(get_clockwise_neighbors(p1, tuple(points_neighbors[p1])))
        if len(points_neighbors[p2]) == 2:
          points_neighbors[p2] = list(get_clockwise_neighbors(p2, tuple(points_neighbors[p2])))
        virtual_edges.append(self.sort_points(edge.points()))
    
    neighbors_edges_pairs = [(points_neighbors, virtual_edges)]

    for point in self.points:
      min_dist, min_elem, min_inter = self.min_dist_and_elem(point, neighbors_edges_pairs)
      if min_elem != None:
        elems_to_points[min_elem].append(point)
        dist_to_pairs.put((min_dist, Pair(point, min_elem, min_inter)))
        points_to_elem[point] = min_elem


    if self.is_debug: print("START")
    marked_for_delete_points, marked_for_delete_polys = [], []
    while not dist_to_pairs.empty():
      dist, pair = dist_to_pairs.get()
      point, elem, inter = pair.point, pair.elem, pair.inter
      if point not in marked_for_delete_points and points_to_elem[point] == elem:
        point_other, point_others = None, []
        if type(elem) != hou.Point:
          # Point to Edge Contraction
          if self.is_debug: print("P-E: " + str((point.number(), (elem[0].number(), elem[1].number()))))
          for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
            if point in points_neighbor_ and elem[0] in points_neighbor_ and elem[1] in points_neighbor_:
              points_neighbors = points_neighbor_
              virtual_edges = virtual_edges_
          neighbors_edges_pairs.remove((points_neighbors, virtual_edges))
          '''
              Typical Point-Edge Contraction
                    elem
              er____________el       er____     _____el
                                =>        \   /
                                        point
              pl___point____pr       pl____/   \_____pr

              Edge case Point-Edge Contraction
                  
                  el                              el
                /                               /
                / elem           =>  other----point
              /                                  \ 
              other___point___pr                   pr

          '''
          point_new_position = (inter + point.position()) / 2
          point_movement = point_new_position - point.position()
          elem_movement = point_new_position - inter
          point.setPosition(point_new_position)
          for prim in point.prims():
            if prim.type() == hou.primType.Polygon and prim not in marked_for_delete_polys:
              for prim_point in prim.points():
                if prim_point not in points:
                  neighbors_pull[prim_point] = (neighbors_pull_weight * point_movement) if prim_point not in neighbors_pull else (neighbors_pull[prim_point] + neighbors_pull_weight * point_movement)
                  neighbors_pull_count[prim_point] += 1
                  
          elem_l, elem_r = elem[0] if points_neighbors[elem[0]][1] == elem[1] else elem[1], elem[0] if points_neighbors[elem[0]][0] == elem[1] else elem[1]
          elem_polys = set(elem_l.prims()).intersection(set(elem_r.prims()))
          elem_poly = None
          while len(elem_polys) > 0:
            curr_poly = elem_polys.pop()
            if curr_poly not in marked_for_delete_polys:
              elem_poly = curr_poly
          assert(elem_poly != None)
          marked_for_delete_polys.append(elem_poly)
          poly_points = elem_poly.points()
          
          poly_1, poly_2 = poly_points.copy(), poly_points.copy()
          elem_r_index, elem_l_index = poly_1.index(elem_r), poly_2.index(elem_l)
          poly_1[elem_r_index], poly_2[elem_l_index] = point, point
          geo.createPolygons((tuple(poly_1), tuple(poly_2)))

          old_elem_edges = ([self.sort_points((elem_l, p)) for p in points_neighbors[elem_l] if p != elem_r] 
                          + [self.sort_points((elem_r, p)) for p in points_neighbors[elem_r] if p != elem_l])
          old_point_edges = [self.sort_points((point, p)) for p in points_neighbors[point]]
          duplicate_edges = set(old_point_edges).intersection(set(old_elem_edges))
          affected_elems = list(virtual_edges) + list(points_neighbors.keys())

          points_neighbors_l, points_neighbors_r = defaultdict(list), defaultdict(list)
          virtual_edges_l, virtual_edges_r = [], []
          point_l, point_r = points_neighbors[point]
          points_to_elem[point] = None

          points_neighbors[point] = [point_l, elem_r]
          points_neighbors[elem_r][0] = point
          curr, is_loop = point, False
          while not is_loop:
            points_neighbors_l[curr] = points_neighbors[curr]
            virtual_edges_l.append(self.sort_points((curr, points_neighbors[curr][0])))
            curr = points_neighbors[curr][0]
            if curr == point:
              is_loop = True
          
          points_neighbors[point] = [elem_l, point_r]
          points_neighbors[elem_l][1] = point
          curr, is_loop = point, False
          while not is_loop:
            points_neighbors_r[curr] = points_neighbors[curr]
            virtual_edges_r.append(self.sort_points((curr, points_neighbors[curr][0])))
            curr = points_neighbors[curr][0]
            if curr == point:
              is_loop = True

          marked_for_delete_groups = []
          new_neighbors_edges_pairs = [(points_neighbors_l, virtual_edges_l), (points_neighbors_r, virtual_edges_r)]
          while len(duplicate_edges) > 0:
            pa, pb = duplicate_edges.pop()
            point_other = pa if pb == point else pb
            points_to_elem[point_other] = None
            if point_other in points_neighbors_l: marked_for_delete_groups.append((points_neighbors_l, virtual_edges_l))
            if point_other in points_neighbors_r: marked_for_delete_groups.append((points_neighbors_r, virtual_edges_r))
            point_others.append(point_other)

          for marked_for_delete_group in marked_for_delete_groups:
            new_neighbors_edges_pairs.remove(marked_for_delete_group)
            del marked_for_delete_group
          if not new_neighbors_edges_pairs:
            points_to_elem[point] = None
          neighbors_edges_pairs += new_neighbors_edges_pairs

        if type(elem) == hou.Point:
          # Point to Point Contraction
          for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
            if point in points_neighbor_ and elem in points_neighbor_:
              points_neighbors = points_neighbor_
              virtual_edges = virtual_edges_
          neighbors_edges_pairs.remove((points_neighbors, virtual_edges))
          is_ee_contraction = self.sort_points((point, elem)) in virtual_edges
          if self.is_debug: print("P-P " + ("EE" if is_ee_contraction else "NE") + ":" + str((point.number(), elem.number())))
          ''' 
              Typical Point-Point Contraction
              p1____elem____p2     p1 _____    _____p2
                                =>        elem               Non-Edge Contraction
              p3____point____p4     p3____/    \____p4
          OR
              p1___point___elem___p2  =>   p1____elem____p2  Edge Contraction

              Edge case Point-Point Contraction
                    / elem        
                   /         =>   other------point           Non-Edge Contraction
                  /            
              other____point
          OR
                    elem
                    /   |
                   /    |    =>   other------point           Edge Contraction
                  /     |
              other____point
          '''
          point_new_position = (elem.position() + point.position()) / 2
          point_movement = point_new_position - point.position()
          elem_movement = point_new_position - elem.position()
          elem.setPosition(point_new_position)
          for prim in point.prims():
            if prim.type() == hou.primType.Polygon and prim not in marked_for_delete_polys:
              for prim_point in prim.points():
                if prim_point not in points:
                  neighbors_pull[prim_point] = (neighbors_pull_weight * point_movement) if prim_point not in neighbors_pull else (neighbors_pull[prim_point] + neighbors_pull_weight * point_movement)
                  neighbors_pull_count[prim_point] += 1
          for prim in elem.prims():
            if prim.type() == hou.primType.Polygon and prim not in marked_for_delete_polys:
              for prim_point in prim.points():
                if prim_point not in points:
                  neighbors_pull[prim_point] = (neighbors_pull_weight * elem_movement) if prim_point not in neighbors_pull else (neighbors_pull[prim_point] + neighbors_pull_weight * elem_movement)
                  neighbors_pull_count[prim_point] += 1
          
          for prim in point.prims():
            if prim.type() == hou.primType.Polygon and prim not in marked_for_delete_polys:
              if elem not in prim.points():
                poly_points = prim.points()
                point_index = poly_points.index(point)
                poly_points[point_index] = elem
                new_poly = geo.createPolygon()
                for poly_point in poly_points:
                  new_poly.addVertex(poly_point)
              marked_for_delete_polys.append(prim)
          marked_for_delete_points.append(point)

          old_elem_edges = [self.sort_points((elem, p)) for p in points_neighbors[elem]]
          old_point_edges = [self.sort_points((point, p)) for p in points_neighbors[point]]
          new_point_edges = [self.sort_points((elem, p)) for p in points_neighbors[point] if p != elem]
          duplicate_edges = set(new_point_edges).intersection(set(old_elem_edges))

          point_l, point_r = points_neighbors[point]
          elem_l, elem_r = points_neighbors[elem]
          points_to_elem[point] = None
          if is_ee_contraction:
            # Edge Contraction
            affected_elems = old_point_edges + old_elem_edges + [point, elem]
            virtual_edges = ((set(virtual_edges) - set(old_point_edges)).union(set(new_point_edges))
                    - duplicate_edges - set([self.sort_points((point, elem))]))
            
            affected_pairs_ = []
            for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
              if point in points_neighbor_:
                affected_pairs_.append((points_neighbor_, virtual_edges_))
            for affected_points_neighbor, affected_virtual_edges in affected_pairs_:
              neighbors_edges_pairs.remove((affected_points_neighbor, affected_virtual_edges))
              old_virtual_edges, new_virtual_edges = [], []
              for neighbor in affected_points_neighbor[point]:
                old_virtual_edges.append(self.sort_points((point, neighbor)))
                new_virtual_edges.append(self.sort_points((elem, neighbor)))
                point_index = affected_points_neighbor[neighbor].index(point)
                affected_points_neighbor[neighbor][point_index] = elem 

              affected_virtual_edges = set(affected_virtual_edges).union(set(new_virtual_edges)) - set(old_virtual_edges)
              affected_elems += old_virtual_edges
              affected_points_neighbor[elem] = affected_points_neighbor[point]

              del affected_points_neighbor[point] 
              neighbors_edges_pairs.append((affected_points_neighbor, affected_virtual_edges))

                
            del points_neighbors[point]
            points_neighbors[point_l][1] = elem if point_l != elem else point_r
            points_neighbors[point_r][0] = elem if point_r != elem else point_l

            while len(duplicate_edges) > 0:
              p1, p2 = duplicate_edges.pop()
              point_other = p1 if p2 == elem else p2
              points_to_elem[p1] = None
              del points_neighbors[p1]
              points_to_elem[p2] = None
              del points_neighbors[p2]
              affected_elems += [point_other]
              point_others.append(point_other)
            neighbors_edges_pairs.append((points_neighbors, virtual_edges))

          else:
            # Non-Edge Contraction
            affected_elems = list(virtual_edges) + list(points_neighbors.keys())
            points_neighbors_l, points_neighbors_r = defaultdict(list), defaultdict(list)
            virtual_edges_l, virtual_edges_r = [], []
            points_neighbors[elem] = [elem_l, point_r]
            points_neighbors[point_r][0] = elem
            curr, is_loop = elem, False
            while not is_loop:
              points_neighbors_l[curr] = points_neighbors[curr]
              virtual_edges_l.append(self.sort_points((curr, points_neighbors[curr][0])))
              curr = points_neighbors[curr][0]
              if curr == elem:
                is_loop = True

            points_neighbors[elem] = [point_l, elem_r]
            points_neighbors[point_l][1] = elem
            curr, is_loop = elem, False
            while not is_loop:
              points_neighbors_r[curr] = points_neighbors[curr]
              virtual_edges_r.append(self.sort_points((curr, points_neighbors[curr][0])))
              curr = points_neighbors[curr][0]
              if curr == elem:
                is_loop = True

            marked_for_delete_groups = []
            new_neighbors_edges_pairs = [(points_neighbors_l, virtual_edges_l), (points_neighbors_r, virtual_edges_r)]
            while len(duplicate_edges) > 0:
              pa, pb = duplicate_edges.pop()
              point_other = pa if pb == elem else pb
              points_to_elem[point_other] = None
              if point_other in points_neighbors_l: marked_for_delete_groups.append((points_neighbors_l, virtual_edges_l))
              if point_other in points_neighbors_r: marked_for_delete_groups.append((points_neighbors_r, virtual_edges_r))
              point_others.append(point_other)

            for marked_for_delete_group in marked_for_delete_groups:
              new_neighbors_edges_pairs.remove(marked_for_delete_group)
              del marked_for_delete_group
            if not new_neighbors_edges_pairs:
              points_to_elem[elem] = None
            neighbors_edges_pairs += new_neighbors_edges_pairs
        
        affected_elems = list(set(affected_elems))
        affected_elems_points = []
        for affected_elem in affected_elems:
          for affected_elem_point in elems_to_points[affected_elem]:
            if affected_elem_point not in point_others:
              affected_elems_points.append(affected_elem_point)
          del elems_to_points[affected_elem]
        for affected_elems_point in affected_elems_points:
          points_to_elem[affected_elems_point] = None
          if affected_elems_point not in marked_for_delete_points and affected_elems_point not in point_others:
            min_dist, min_elem, min_inter = self.min_dist_and_elem(affected_elems_point, neighbors_edges_pairs)
            if min_elem != None:
              elems_to_points[min_elem].append(affected_elems_point)
              points_to_elem[affected_elems_point] = min_elem
              dist_to_pairs.put((min_dist, Pair(affected_elems_point, min_elem, min_inter)))

        '''for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
          temp_list = []
          print("PN, VE")
          for key, value in points_neighbor_.items():
            vs = []
            for v in value:
              vs.append(v.number())
            temp = [key.number(), vs]
            temp_list.append(temp)
          print(temp_list)
          temp_list = []
          for virtual_edge in virtual_edges_:
            temp_list.append((virtual_edge[0].number(), virtual_edge[1].number()))
          print(temp_list)
            
        temp_list = []
        print("points_to_elem")
        for key, value in points_to_elem.items():
          if type(value) == hou.Point:
            elem = value.number()
          elif value == None:
            elem = None
          else:
            elem = (value[0].number(), value[1].number())
          temp_list.append([key.number(), elem])
        print(temp_list)'''
    for prim_point, point_movement in neighbors_pull.items():
      prim_point.setPosition((prim_point.position() + point_movement / (neighbors_pull_count[prim_point])))
    return marked_for_delete_polys, marked_for_delete_points

# ------------ Hole-Filling Classes ------------ ~
class Island_Fill():
  def __init__(self, geo, points, inner_points, is_min_tri=False, is_bounded=True):
    self.geo = geo
    self.points = points
    self.inner_points = inner_points
    self.is_min_tri = is_min_tri
    self.is_bounded = is_bounded

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

    marked_for_delete_polys, marked_for_delete_points = [], []
    if not self.is_bounded:
      p_1_to_p_2_outer = np.append(np.array(points_outer[:p_2_outer_ind]), [p_2_outer])
      p_2_to_p_1_outer = np.append(np.array(points_outer[p_2_outer_ind:]), [p_1_outer])
      p_1_to_p_2_inner = np.append(np.array(points_inner[:p_2_inner_ind]), [p_2_inner])
      p_2_to_p_1_inner = np.append(np.array(points_inner[p_2_inner_ind:]), [p_1_inner])

      p_1_to_p_2_inner_rev = p_1_to_p_2_inner[::-1]
      p_2_to_p_1_inner_rev = p_2_to_p_1_inner[::-1]
      boundaries = [(p_1_to_p_2_outer, p_1_to_p_2_inner_rev), (p_2_to_p_1_outer, p_2_to_p_1_inner_rev)] # outer, inner
    else:
      p_1_to_p_2_prev_outer = np.array(points_outer[:p_2_outer_ind])
      p_2_to_p_1_prev_outer = np.array(points_outer[p_2_outer_ind:])
      p_1_to_p_2_prev_inner = np.array(points_inner[:p_2_inner_ind])
      p_2_to_p_1_prev_inner = np.array(points_inner[p_2_inner_ind:])

      p_1_poly_outer = [p_2_to_p_1_prev_outer[len(p_2_to_p_1_prev_outer) - 1], p_1_outer, p_1_inner]
      p_1_poly_inner = [p_1_inner, p_2_to_p_1_prev_inner[len(p_2_to_p_1_prev_inner) - 1], p_2_to_p_1_prev_outer[len(p_2_to_p_1_prev_outer) - 1]]
      p_2_poly_outer = [p_1_to_p_2_prev_outer[len(p_1_to_p_2_prev_outer) - 1], p_2_outer, p_2_inner]
      p_2_poly_inner = [p_2_inner, p_1_to_p_2_prev_inner[len(p_1_to_p_2_prev_inner) - 1], p_1_to_p_2_prev_outer[len(p_1_to_p_2_prev_outer) - 1]]
      
      self.geo.createPolygons((p_1_poly_outer, p_1_poly_inner, p_2_poly_outer, p_2_poly_inner))

      p_1_to_p_2_prev_inner_rev = p_1_to_p_2_prev_inner[::-1]
      p_2_to_p_1_prev_inner_rev = p_2_to_p_1_prev_inner[::-1]
      boundaries = [(p_1_to_p_2_prev_outer, p_1_to_p_2_prev_inner_rev), (p_2_to_p_1_prev_outer, p_2_to_p_1_prev_inner_rev)] # outer, inner

    '''
    D. We can split the two regular holes to more holes, depending on the size of the hole
        Fill the holes using any method
    '''
    max_boundary_size = 200
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
        if not self.is_bounded:
          inner_1 = np.append(inner[:inner_ind], [inner_point])
          inner_2 = inner[inner_ind:]
          outer_1 = outer[outer_ind:]
          outer_2 = np.append(outer[:outer_ind], [outer_point])
          boundaries.append((outer_1, inner_1))
          boundaries.append((outer_2, inner_2))
        else:
          inner_1 = inner[:inner_ind]
          inner_2 = inner[inner_ind:]
          outer_1 = outer[outer_ind:]
          outer_2 = outer[:outer_ind]
          poly_1 = [inner_point, outer_point, inner_1[len(inner_1) - 1]]
          poly_2 = [outer_point, inner_point, outer_2[len(outer_2) - 1]]
          self.geo.createPolygons((poly_1, poly_2))
          boundaries.append((outer_1, inner_1))
          boundaries.append((outer_2, inner_2))
      else:
        num_split += 1
        if self.is_min_tri:
          MinTriangulation(self.geo, np.append(outer, inner)).min_triangulation(generate=True)
        else:
          marked_for_delete_polys_, marked_for_delete_points_ = GapContraction(self.geo, np.append(outer, inner)).fill()
          self.geo.deletePrims(marked_for_delete_polys_, keep_points=True)
          #marked_for_delete_polys += [p for p in marked_for_delete_polys_ if p not in marked_for_delete_polys]
          marked_for_delete_points += [p for p in marked_for_delete_points_ if p not in marked_for_delete_points]
    #self.geo.deletePrims(marked_for_delete_polys, keep_points=True)
    self.geo.deletePoints(marked_for_delete_points)

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
  # NOTE: Minimum triangulation can be either bounded or not, gap contraction must be bounded
  Island_Fill(geo, points, patch_points, is_min_tri=False, is_bounded=True).fill()

node.bypass(True)


