import numpy as np
node = hou.pwd()
geo = node.geometry()
boundary_node = node.inputs()[1]
boundaries = boundary_node.geometry().pointGroups()
edge_boundary_node = node.inputs()[2]

edge_groups = edge_boundary_node.parm("group1").eval()
target_edge_groups = ""
for i in range(len(boundaries) - 1):
  target_edge_groups += "group1__" + str(i) + " "
target_edge_groups = target_edge_groups[:-1]
if not target_edge_groups == edge_groups:
  edge_boundary_node.parm("group1").set(target_edge_groups)

edge_boundaries = edge_boundary_node.geometry().edgeGroups()
'''
4. Optimize Camera View Proportions by:
- Ensuring equitable locality knowledge in vertical and horizontal axes 
'''
pix = geo.findPointAttrib("pix")
path_name = hou.hipFile.name().split(".")[0]
if (pix):
  resolutions_x = []
  resolutions_y = []
  for i in range(1, len(boundaries)):
    boundary = boundaries[i]
    edge_boundary = edge_boundaries[i-1]
    points = boundary.points()
    edges = edge_boundary.edges()
    min_x = float('inf')
    max_x = 0
    min_y = float('inf')
    max_y = 0
    for point in points:
      pix_attrib = point.attribValue(pix)
      curr_x = pix_attrib[(i-1) * 3]
      curr_y = pix_attrib[(i-1) * 3 + 1]
      min_x = min(min_x, curr_x)
      max_x = max(max_x, curr_x)
      min_y = min(min_y, curr_y)
      max_y = max(max_y, curr_y)
    x_len = max_x - min_x
    y_len = max_y - min_y
    camera = hou.node('/obj/oz_camera_' + str(i))
    x_prior_res = camera.parm('resx').eval()
    y_prior_res = camera.parm('resy').eval()
    x_prop_res = x_prior_res * (x_len/x_prior_res)
    y_prop_res = y_prior_res * (y_len/y_prior_res)
    '''
    5. Optimize Camera Resolution by:
    - Maximizes retainment of 3D information in 2D
    - Minimizes rendering time
    We approx S Salamanca,P Merchan,A Adan,E Perez,C Cerrada[2008],
    resolution = n * m
    where
      n = number of width pixels = x_prop_res/q
      m = number of height pixels = y_prop_res/q
      q = alpha * d
      where
        d = mesh edge average length (Take this locally)
        alpha = 0.87, predetermined by finding Optimal Occupation Ratio,
        optimal(Oi) = max(Oi) = max(SUMi_1,n SUMj_1,m Vij / n*m)
        where
                  | 1 iff pixel pij taken up optimally
            Vij = | 0 otherwise
    '''
    d = 0
    for edge in edges:
        d += edge.length()
    d /= len(edges)
    x_true_res = int(x_prop_res / (0.87 * d))
    y_true_res = int(y_prop_res / (0.87 * d))
    camera.parm('resx').set(x_true_res)
    camera.parm('resy').set(y_true_res)
    resolutions_x.append(x_true_res)
    resolutions_y.append(y_true_res)

  if (not(geo.findGlobalAttrib("resolutionsx_new") or geo.findGlobalAttrib("resolutionsy_new"))):
    geo.addAttrib(hou.attribType.Global, "resolutionsx_new", resolutions_x)
    geo.addAttrib(hou.attribType.Global, "resolutionsy_new", resolutions_y)
  else:
    geo.setGlobalAttribValue("resolutionsx_new", resolutions_x)
    geo.setGlobalAttribValue("resolutionsy_new", resolutions_y)
node.bypass(True)