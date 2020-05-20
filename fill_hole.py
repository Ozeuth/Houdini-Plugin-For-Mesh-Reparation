import math
import numpy as np
node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()

# Choose 3D Context Region
boundaries = inputs[1].geometry().pointGroups()
i = 0
clean = []
for boundary in boundaries:
    if i == 0:
      i += 1
      continue
    '''
    1. Compute a Projection Plane via Least Squares
    Plane eqn: ax + by + c = z
         A        x   =   b
    | x0 y0 1 |         | z0 |
    | x1 y1 1 | | a |   | z1 |  => | a |
    | ....... | | b | = | .. |     | b | = (A^T*A)^-1*A^T*b
    | xn yn 1 | | c |   | zn |     | c |
    '''
    A = []
    b = []
    boundary_center = np.array([0, 0, 0])
    points = boundary.points()
    for point in points:
      point_pos = point.position()
      A.append([point_pos[0], point_pos[1], 1])
      b.append(point_pos[2])
      boundary_center = boundary_center + np.array(point_pos)
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
    '''
    2. Camera is fit to Projection Plane via translation + rotation
    '''
    if (hou.node('/obj/oz_camera_' + str(i)) != None):
      camera = hou.node('/obj/oz_camera_' + str(i))
    else:
      camera = hou.node('/obj').createNode('cam', 'oz_camera_' + str(i))
    if (not fit_fail):
      plane_normal = np.array([-1 * a/c, -1 * b/c, 1/c])
    else:
      plane_normal = np.array([0.001, 1, 0.001]) * (1 if ((a * b * c) >= 0) else -1)
    translation = hou.Matrix4((1, 0, 0, boundary_center[0],
                               0, 1, 0, boundary_center[1],
                               0, 0, 1, boundary_center[2], 
                               0, 0, 0, 1)).transposed()
    v = math.sqrt(math.pow(plane_normal[0], 2) + math.pow(plane_normal[2], 2))
    rotation_y = hou.Matrix4((plane_normal[2]/v, 0, -1 * plane_normal[0]/v, 0,
                              0, 1, 0, 0,
                              plane_normal[0]/v, 0, plane_normal[2]/v, 0, 
                              0, 0, 0, 1))
    d = math.sqrt(math.pow(plane_normal[0], 2) + math.pow(plane_normal[1], 2) + math.pow(plane_normal[2], 2))
    rotation_x = hou.Matrix4((1, 0, 0, 0,
                            0, v/d,  -1 * plane_normal[1]/d, 0,
                            0, plane_normal[1]/d, v/d, 0, 
                            0, 0, 0, 1))
    camera.setWorldTransform(rotation_x * rotation_y * translation)
    camera.parm('resx').set(1920)
    camera.parm('resy').set(1080)
    '''
    3. Camera is zoomed out until it views all boundary point
      A point is viewable if it is given a valid UV coordinate
      (uv_x, uv_y where 0 <= uv_x, uv_y <= 1) when the mesh is
      unwrapped using camera perspective

      While not all boundary points are viewable,
      We zoom the camera out, and re-unwrap and check again
    '''
    if (hou.node(hou.parent().path() + "/oz_uvtexture_" + str(i)) != None):
      uv_plane = hou.node(hou.parent().path() + "/oz_uvtexture_" + str(i))
    else:
      uv_plane = node.parent().createNode('texture', 'oz_uvtexture_' + str(i))
    for child in hou.parent().children():
      if (child.type().name() == "file"):
        file_node = child
        break
    uv_plane.setInput(0, file_node)
    uv_plane.parm("type").set(9)
    uv_plane.parm("campath").set(camera.path())
    uv_plane.parm("coord").set(0)
    uv_points = uv_plane.geometry().points()

    visible_points = 0
    max_visible_points = len(points)
    scale = 0
    # BUG: There may exist no camera view where all boundary points are visible. Fix Later
    while (visible_points != max_visible_points):
      visible_points = 0
      for point in points:
        for uv_point in uv_points:
          uv_coord = uv_point.attribValue("uv")
          if point.number() == uv_point.number() and uv_coord[0] >= 0 and uv_coord[0] <= 1 and uv_coord[1] >= 0 and uv_coord[1] <= 1:
            visible_points += 1
      if (visible_points < max_visible_points):
        scale += 0.1
        camera_normal = plane_normal * scale
        new_translation = hou.Matrix4((1, 0, 0, boundary_center[0] + plane_normal[0],
                               0, 1, 0, boundary_center[1] + plane_normal[1],
                               0, 0, 1, boundary_center[2] + plane_normal[2], 
                               0, 0, 0, 1)).transposed()
        camera.setWorldTransform(rotation_x * rotation_y * new_translation)
    i += 1
    '''
    o_bbox = geo.boundingBox()
    o_bbox.setTo((0, 0, 0, 0, 0, 0))
    for point in boundary.points():
        o_bbox.enlargeToContain(point.position())
    c_min = o_bbox.minvec() * 2.5
    c_max = o_bbox.maxvec() * 2.5
    c_bbox = geo.boundingBox()
    c_bbox.setTo((c_min[0], c_min[1], c_min[2], c_max[0], c_max[1], c_max[2]))

    for point in boundary.points(): # For every point in the boundary
      for vertex in point.vertices(): # For every vertex in the point
        if (vertex.prim().type() == hou.primType.Polygon):
            vertex.prim().points() # We get the other points in the vertex's primitive

            # If all points are in the bounding box, we add it in
  '''