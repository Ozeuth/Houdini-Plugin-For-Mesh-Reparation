import hou
import numpy as np
from hausdorff import hausdorff_distance

node = hou.pwd()
geo = node.geometry()

merge_nodes = hou.session.find_nodes("oz_merge_")

point_boundaries = []
point_patchs = []
for point_group in geo.pointGroups():
  if "patch" in point_group.name():
    point_patchs.append(point_group)
  else:
    point_boundaries.append(point_group)


for i, merge_node in enumerate(merge_nodes):
  points = point_boundaries[i].points()
  points_patch = point_patchs[i].points()

  lo_node = merge_node.inputs()[0]
  hi_node = merge_node.inputs()[1]
  
  lo_points = lo_node.geometry().points()
  hi_points = hi_node.geometry().points()
  '''
  1. hi-freq patches are of different scale from original lo-freq patches
     Haudorff distance computes the "similarity" of the patches
     We can thus compute an ideal scale by an evolutionary algorithm on min Hausdorff distances
  '''
  lo_points_pos = []
  for lo_point in lo_points:
    lo_points_pos.append(lo_point.position())
  lo_points_pos = np.array(lo_points_pos)

  hi_points_pos = []
  for hi_point in hi_points:
    hi_points_pos.append(hi_point.position())
  hi_points_pos = np.array(hi_points_pos)

  best_scale, best_dist = 0.15, float('inf')
  min_scale, max_scale = 0, 1
  tries, max_tries = 0, 10
  sample_size = 15
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

  print("Scaled hi-freq patch by " + str(best_scale) + " to lo-freq patch size, with error " + str(best_dist))
  #print(lo_node.parms())
  #print(lo_node.parm("ty").eval())
  print(hou.session.find_parm(lo_node, "tx"))
  print(hou.session.find_parm(lo_node, "tz"))
  print(hou.session.find_parm(lo_node, "ty"))
  #lo_node_translate = hou.Vector3((hou.session.find_parm(lo_node, "tx"), hou.session.find_parm(lo_node, "ty"), hou.session.find_parm(lo_node, "tz")))
  for point in points_patch:
    point.setPosition(point.position() * best_scale - lo_node_translate)
