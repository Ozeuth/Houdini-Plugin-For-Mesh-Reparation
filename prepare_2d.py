import PIL
import hou
import math
import numpy as np
import sys
from PIL import Image, ImageDraw

node = hou.pwd()
geo = node.geometry()
boundary_node = node.inputs()[1]
boundaries = boundary_node.geometry().pointGroups()
edge_boundary_node = node.inputs()[2]
edge_boundaries = edge_boundary_node.geometry().edgeGroups()

is_alpha = bool(hou.session.find_parm(hou.parent(), "isAlpha"))
'''
 8. Map 3D Context Region -> 2D Render
  Generate Mask and Conditioning Area combined image from mapping,
  ALL_x,y mcw[x, y] = (255, 0, 0)*1_M + (0, 255, 0)*1_C + (0, 0, 255)*1_W + (0, 0, 0)
  where
    1_M = 1 if in Mask else 0
    1_C = 1 if in Conditioning Area else 0
    1_W = 1 if in subdomain W else 0
  [Optional]: We determine Information Loss from mapping (Optimal Occupancy Ratio)
'''
pix = geo.findPointAttrib("pix_new")
path_name = hou.hipFile.name().split(".")[0]
if (pix):
  for i in range(1, len(boundaries)):
    boundary = boundaries[i]
    edge_boundary = edge_boundaries[i-1]
    points = boundary.points()
    edges = edge_boundary.edges()

    pix_pos = []
    for point in points:
      pix_attrib = point.attribValue(pix)
      pix_point = (pix_attrib[(i-1) * 3], pix_attrib[(i-1) * 3 + 1], pix_attrib[(i-1) * 3 + 2])
      pix_pos.append((point.number(), pix_point))
    pix_pos = dict(pix_pos)

    # Optionally check Optimal Occupancy Ratio
    if (is_alpha):
      uv_all_node = hou.node(hou.parent().path() + "/uv_viewer_all")
      camera = hou.node('/obj/oz_camera_' + str(i))
      x_res = camera.parm("resx").eval()
      y_res = camera.parm("resy").eval()
      uv_all_node.parm("campath").set(camera.path())
      all_geo = node.inputs()[3].geometry()

      pix_ratio = all_geo.findPointAttrib("pix_ratio")
      all_points = all_geo.points()
      pix_all_num = {}
      for all_point in all_points:
        uv_coord = all_point.attribValue("uv_all")
        # TODO: Do not include points hidden behind faces. Rop Sop
        if (uv_coord[0] >= 0 and uv_coord[0] <= 1 and uv_coord[1] >= 0 and uv_coord[1] <= 1 and not all(v == 0 for v in uv_coord)):
          pix_all_attrib = all_point.attribValue(pix_ratio)
          pix_all_point = (pix_all_attrib[(i-1) * 3], pix_all_attrib[(i-1) * 3 + 1], pix_all_attrib[(i-1) * 3 + 2])
          pix_approx_point = (int(pix_all_point[0]), int(pix_all_point[1]))
          if not str(pix_approx_point) in pix_all_num:
            pix_all_num[str(pix_approx_point)] = 1
          else:
            pix_all_num[str(pix_approx_point)] += 1
      occupance_ratio = 0
      for x in range(x_res):
        for y in range(y_res):
          if ('(' + str(x) + ', ' + str(y) + ')') in pix_all_num:
            occupance = pix_all_num['(' + str(x) + ', ' + str(y) + ')']
            if occupance == 1:
              occupance_ratio += 1
      occupance_ratio /= (x_res * y_res * 1.0)
      print("Current Occupance Ratio: " + str(occupance_ratio))

    w = 3
    image = Image.open(path_name + "/opening_" + str(i) + ".png")
    image_size = image.size
    image.close()
    mcw = Image.new('RGB', image_size, color=(0, 0, 255))
    draw = ImageDraw.Draw(mcw)
    edge_pixels = []
    for edge in edges:
      edge_points = edge.points()
      edge_pixel_0 = pix_pos.get(edge_points[0].number())
      edge_pixel_0_x = int(edge_pixel_0[0])
      edge_pixel_0_y = int(edge_pixel_0[1])
      edge_pixels.append((edge_pixel_0_x, edge_pixel_0_y))
      draw.ellipse((edge_pixel_0_x - w, edge_pixel_0_y - w, edge_pixel_0_x + w, edge_pixel_0_y + w), fill=(0, 255, 255))
    draw.line(edge_pixels, fill=(0, 255, 255), width= 2*w + 1)
    draw.line((edge_pixels[0], edge_pixels[len(edge_pixels)-1]), fill=(0, 255, 255), width= 2*w + 1)
    draw.polygon(edge_pixels, fill=(255, 0, 0), outline=None)
    mcw.save(path_name + "/mcw_" + str(i) + ".png")
    mcw.close()
  
  '''
  9. 2D inpainting
    We follow B Galerne, A Leclaire[2017],
      inpainting using Gaussian conditional simulation, relying on a Kriging framework
  '''
  # Some Helper functions
  def get_image_het(image):
    image_num = np.sum(np.where(image[:,:,[3]] != 0, 1, 0))
    image_het = [0, 0, 0, 0]
    for d in range(3):
      image_het[d] = np.sum(image[:, :, d])
    image_het = image_het / image_num
    return image_het

  def get_image_t(image, image_het=None):
    t_image = np.zeros(image.shape)
    if not image_het:
      image_het = get_image_het(image)
    t_image[:,:,...] = (np.maximum((image - image_het), t_image) / math.sqrt(image_num)) * 255
    t_image[:,:,[3]] = image[:,:,[3]]
    return t_image    
  
  def convolve2d_fft(A, B):
    C = np.zeros((A.shape[0], A.shape[1], A.shape[2], B.shape[2]))
    f_B = np.zeros((A.shape[0], A.shape[1], B.shape[-1]), dtype=np.complex128)
    for i_M in np.arange(B.shape[-1]):
      f_B[:, :, i_M] = np.fft.fft2(B[:, :, i_M], s=A.shape[:2])
          
    for i_N in np.arange(A.shape[-1]):
      f_A = np.fft.fft2(A[:, :, i_N])
      for i_M in np.arange(B.shape[-1]):
        C[:, :, i_N, i_M] = np.real(np.fft.ifft2(f_A*f_B[:, :, i_M]))

    if (B.shape[2] == 1):
      return C.reshape((A.shape[0], A.shape[1], A.shape[2]))
    else:
      return C
  
  def CGD(phi, c, k_max=1000, epsilon=0.01):
    A = np.ma.masked_where(np.where(c[:,:,[3]] != 0, 1, 0), math.pow(c-get_image_het(c), 2)).filled(fill_value=0)
    k, psi  = 0, 0
    r = np.matmul(np.transpose(A), phi) - np.matmul(np.transpose(A), A) * psi
    d = r
    return None

  for i in range(1, len(boundaries)):
    image = Image.open(path_name + "/opening_" + str(i) + ".png")
    image = np.array(image)
    image_size = image.shape
    mcw = Image.open(path_name + "/mcw_" + str(i) + ".png")
    mcw = np.array(mcw)
    '''
    9A. Compute
      v_het = 1/|w|*[SUMr_elem(w)(v(r)), SUMg_elem(g)(v(g)), SUMb_elem(b)(v(b))]
            | 1/sqrt(|w|)*(v-v_het)
      t_v = | 0 otherwise
        where
          v = w restricted to u (w where u is opaque)
    '''
    v = np.ma.masked_where(np.where(np.logical_and(mcw[:,:,[2]] == 255, image[:,:,[3]] > 0.001), [1, 1, 1, 1], [0, 0, 0, 0])==0, image).filled(fill_value=0)
    v_het = get_image_het(v)
    t_v = get_image_t(v)
    '''
    9B. Draw Gaussian Sample,
      F = convolve(t_v, W)
      where
        W = Gaussian White Noise
    '''
    # TODO: Check if W is meant to be size of image, or a small window
    # TODO: Convolution should ignore pixels where alpha == 0
    W = np.random.normal(0, 1, (image_size[0], image_size[1], 1)).astype(np.float32)
    F = convolve2d_fft(t_v, W)
    '''
    9C. Compute using GCD
      psi_1 = gamma_t |cxc (v - v_het)
      psi_2 = gamma_t |cxc (F|c)
    '''
    c = np.ma.masked_where(np.where(np.logical_and(mcw[:,:,[1]] == 255, image[:,:,[3]] > 0.001), [1, 1, 1, 1], [0, 0, 0, 0])==0, image).filled(fill_value=0)
    
    if i ==1:
      im = Image.fromarray(np.uint8(c))
      im.save(path_name + "/test.png")
      im.close()
node.bypass(True)