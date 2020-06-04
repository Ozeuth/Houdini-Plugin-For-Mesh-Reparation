import numpy as np
import numpy.random as random

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
uv_node = hou.node(hou.parent().path() + "/uv_viewer")

cameras_info = hou.session.cameras_info
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
    x_prop_res = max_x - min_x
    y_prop_res = max_y - min_y

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
    q = 0.87 * d
    x_true_res = x_prop_res / q
    y_true_res = y_prop_res / q
    # NOTE: Houdini Non-Commercial limited to 1280 x 720, so scale res down
    downscale = 1
    if (x_true_res > 1280 or y_true_res > 720):
      x_downscale = x_true_res / 1280
      y_downscale = y_true_res / 720
      downscale = max(x_downscale, y_downscale)
    x_true_res = int(x_true_res / downscale)
    y_true_res = int(y_true_res / downscale)

    camera = hou.node('/obj/oz_camera_' + str(i))
    camera.parm('resx').set(x_true_res)
    camera.parm('resy').set(y_true_res)
    resolutions_x.append(x_true_res)
    resolutions_y.append(y_true_res)

    '''
    6. Camera is zoomed to minimum zoom where all boundary points are visible.
    While not all boundary points are viewable, We zoom the camera
    to a value within the zoom range (converging on ideal values),
    re-unwrap and check again
    '''
    uv_node.parm("campath").set(camera.path())
    boundary_center = cameras_info["centers"][i-1]
    rotation_x = cameras_info["rotationsx"][i-1]
    rotation_y = cameras_info["rotationsy"][i-1]
    plane_normal = cameras_info["normals"][i-1]
    zoom_out = cameras_info["zooms"][i-1]
      
    zoom_range_max = zoom_out
    best_zoom_out = zoom_out
    best_visible_points = 0
    max_visible_points = len(points)
    tries = 0
    sample_size = 10
    max_tries = 10
    better_zoom_findable = True
    while ((best_visible_points != max_visible_points or better_zoom_findable) and tries < max_tries):
      visible_points_samples = []
      zoom_out_samples = []
      for sample in range(sample_size):
        if sample == 0:
          zoom_out = best_zoom_out
        else:
          zoom_out = random.uniform(0, zoom_range_max)
        visible_points = 0
        camera_normal = plane_normal * zoom_out
        new_translation = hou.Matrix4((1, 0, 0, boundary_center[0] + camera_normal[0],
                              0, 1, 0, boundary_center[1] + camera_normal[1],
                              0, 0, 1, boundary_center[2] + camera_normal[2], 
                              0, 0, 0, 1)).transposed()
        hou.session.reset_camera(camera)
        camera.setWorldTransform(rotation_x * rotation_y * new_translation)
        for point in points:

          uv_coord = point.attribValue("uv")
          if (uv_coord[0] >= 0 and uv_coord[0] <= 1 and uv_coord[1] >= 0 and uv_coord[1] <= 1 and not all(v == 0 for v in uv_coord)):
            visible_points+= 1
        visible_points_samples.append(visible_points)
        zoom_out_samples.append(zoom_out)

      best_visible_points = max(visible_points_samples)
      viable_zoom_outs = []
      for sample in range(sample_size):
        if (visible_points_samples[sample] == best_visible_points):
          viable_zoom_outs.append(zoom_out_samples[sample])
      best_zoom_out = min(viable_zoom_outs)

      if (best_visible_points == max_visible_points):
        if (zoom_range_max == best_zoom_out):
          better_zoom_findable = False
        zoom_range_max = best_zoom_out
      tries += 1
    if (best_visible_points != max_visible_points): 
      print("WARNING: not all points could be fit to the camera")
    camera_normal = plane_normal * best_zoom_out
    new_translation = hou.Matrix4((1, 0, 0, boundary_center[0] + camera_normal[0],
                          0, 1, 0, boundary_center[1] + camera_normal[1],
                          0, 0, 1, boundary_center[2] + camera_normal[2], 
                          0, 0, 0, 1)).transposed()
    hou.session.reset_camera(camera)
    camera.setWorldTransform(rotation_x * rotation_y * new_translation) 

  if (not(geo.findGlobalAttrib("resolutionsx_new") or geo.findGlobalAttrib("resolutionsy_new"))):
    geo.addAttrib(hou.attribType.Global, "resolutionsx_new", resolutions_x)
    geo.addAttrib(hou.attribType.Global, "resolutionsy_new", resolutions_y)
  else:
    geo.setGlobalAttribValue("resolutionsx_new", resolutions_x)
    geo.setGlobalAttribValue("resolutionsy_new", resolutions_y)
node.bypass(True)