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
    image_arr = np.array(image)
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

    # Clear "void" pixel values, values on pixels that have zero alpha but non-zero rgb
    mcw = np.array(mcw)
    image = image_arr
    mcw_alpha = np.ma.masked_where(np.where(np.logical_and(mcw[:,:,0] < 0.01, np.array(image[:,:,3]) < 0.01), 1, 0), 
      np.ones((mcw.shape[0], mcw.shape[1])) * 255).filled(fill_value=0)
    mcw_alpha_stack = np.stack((mcw_alpha, mcw_alpha, mcw_alpha), axis=-1)
    mcw = np.ma.masked_where(np.where(mcw_alpha_stack==0, 1, 0)==1, mcw).filled(fill_value=0)
    mcw = np.dstack((mcw, mcw_alpha))
    image = np.ma.masked_where(np.where(image[:,:,[3]] == 0, [1, 1, 1, 1], [0, 0, 0, 0])==1, image).filled(fill_value=0)

    mcw = Image.fromarray(np.uint8(mcw))
    mcw.save(path_name + "/mcw_" + str(i) + ".png")
    image = Image.fromarray(np.uint8(image))
    image.save(path_name + "/opening_" + str(i) + ".png")
  '''
  9. 2D inpainting
    We follow B Galerne, A Leclaire[2017],
      inpainting using Gaussian conditional simulation, relying on a Kriging framework
  '''
  # Some Helper functions
  def get_image_num(image, alpha_dim = None):
    if alpha_dim:
      image_num = np.sum(np.where(image[:,:,[alpha_dim]] != 0, 1, 0))
    else:
      image_num = (image.size)
    return image_num

  def get_image_het(image, image_dim, alpha_dim = None):
    image_het = np.zeros(image_dim)
    for d in range(image_dim):
      image_het[d] = np.sum(image[:, :, d])
    image_het = image_het / get_image_num(image, alpha_dim)
    return image_het

  # alpha_dim = 3
  def get_image_t(image, image_dim, image_het=None, alpha_dim=None):
    t_image = np.zeros(image.shape)
    if image_het == None:
      image_het = get_image_het(image, image_dim)
    t_image[:,:,...] = (image - image_het) / math.sqrt(get_image_num(image, alpha_dim))
    if alpha_dim:
      t_image[:,:,[alpha_dim]] = image[:,:,[alpha_dim]]
      t_image = np.where(t_image[:,:,[alpha_dim]] == 0, [0] * image_dim, t_image)
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

  def CGD(phi, A, k_max=1000, epsilon=0.01):
    k, psi  = 0, 0
    r = np.matmul(np.transpose(A), phi) - np.matmul(np.transpose(A), A) * psi
    d = r
    r_curr = np.linalg.norm(r, 2)
    while (r_curr > epsilon and k <= k_max):
      alpha = math.pow(r_curr, 2) / np.matmul(np.transpose(d), np.matmul(np.transpose(A), np.matmul(A, d)))
      psi = psi + np.matmul(alpha , d)
      r = r - np.matmul(alpha, np.matmul(np.transpose(A), np.matmul(A, d)))
      d = r + (math.pow(np.linalg.norm(r, 2), 2) / math.pow(r_curr, 2)) * d
      k += 1
      r_curr = np.linalg.norm(r, 2)
    return psi

  def debug(path_name, i, image_dim, v=None, t_v=None, F=None, cor_t_v=None, c=None, A=None, phi_1=None, phi_2=None, psi_1=None, psi_2=None, kriging_comp=None, innov_comp=None, result=None):
    if image_dim == 1:
      if v != None: v = v.reshape((v.shape[0], v.shape[1]))
      if t_v != None: t_v = t_v.reshape((t_v.shape[0], t_v.shape[1]))
      if F != None: F = F.reshape((F.shape[0], F.shape[1]))
      if cor_t_v != None: cor_t_v = cor_t_v.reshape((cor_t_v.shape[0], cor_t_v.shape[1]))
      if c != None: c = c.reshape((c.shape[0], c.shape[1]))
      if A != None: A = A.reshape((A.shape[0], A.shape[1]))
      if phi_1 != None: phi_1 = phi_1.reshape((phi_1.shape[0], phi_1.shape[1]))
      if phi_2 != None: phi_2 = phi_2.reshape((phi_2.shape[0], phi_2.shape[1]))
      if psi_1 != None: psi_1 = psi_1.reshape((psi_1.shape[0], psi_1.shape[1]))
      if psi_2 != None: psi_2 = psi_2.reshape((psi_2.shape[0], psi_2.shape[1]))
      if kriging_comp != None: kriging_comp = kriging_comp.reshape((kriging_comp.shape[0], kriging_comp.shape[1]))
      if innov_comp != None: innov_comp = innov_comp.reshape((innov_comp.shape[0], innov_comp.shape[1]))
      if result != None: result = result.reshape((result.shape[0], result.shape[1]))
    if v != None:
      im = Image.fromarray(np.uint8(v))
      im.save(path_name + "/" + str(i) + " v.png")
      im.close()
    if t_v != None:
      im = Image.fromarray(np.uint8(t_v))
      im.save(path_name + "/" + str(i) + " t_v.png")
      im.close()
    if F != None:
      im = Image.fromarray(np.uint8(F))
      im.save(path_name + "/" + str(i) + " F.png")
      im.close()
    if cor_t_v != None:    
      im = Image.fromarray(np.uint8(cor_t_v))
      im.save(path_name + "/" + str(i) + " cor_t_v.png")
      im.close()
    if c != None:
      im = Image.fromarray(np.uint8(c))
      im.save(path_name + "/" + str(i) + " c.png")
      im.close()
    if A != None:
      im = Image.fromarray(np.uint8(A))
      im.save(path_name + "/" + str(i) + " A.png")
      im.close()
    if phi_1 != None:
      im = Image.fromarray(np.uint8(phi_1))
      im.save(path_name + "/" + str(i) + " phi_1.png")
      im.close()
    if phi_2 != None:
      im = Image.fromarray(np.uint8(phi_2))
      im.save(path_name + "/" + str(i) + " phi_2.png")
      im.close()
    if psi_1 != None:
      im = Image.fromarray(np.uint8(psi_1))
      im.save(path_name + "/" + str(i) +  " psi_1.png")
      im.close()
    if psi_2 != None:
      im = Image.fromarray(np.uint8(psi_2))
      im.save(path_name + "/" + str(i) + " psi_2.png")
      im.close()
    if kriging_comp != None:
      im = Image.fromarray(np.uint8(kriging_comp))
      im.save(path_name + "/" + str(i) + " kriging.png")
      im.close()
    if innov_comp != None:
      im = Image.fromarray(np.uint8(innov_comp))
      im.save(path_name + "/" + str(i) + " innov.png")
      im.close()
    if result != None:
      im = Image.fromarray(np.uint8(result))
      im.save(path_name + "/" + str(i) + " final.png")
      im.close()

  for i in range(1, len(boundaries)):
    image = Image.open(path_name + "/opening_" + str(i) + ".png").convert('L')
    image = np.array(image)
    image_size = image.shape
    mcw = Image.open(path_name + "/mcw_" + str(i) + ".png")
    mcw = np.array(mcw)

    v = t_v = cor_t_v = c = A = phi_1 = phi_2 = psi_1 = psi_2 = kriging_comp = innov_comp = result = alpha_dim = None
    if len(image.shape) == 2:
      image = image.reshape((image.shape[0], image.shape[1], 1))
      image_dim = 1
    else:
      image_dim = image.shape[len(image.shape) - 1]
    if image_dim == 4:
      alpha_dim = 3
    '''
    9A. Compute
      v_het = 1/|w|*[SUMr_elem(w)(v(r)), SUMg_elem(g)(v(g)), SUMb_elem(b)(v(b))]
            | 1/sqrt(|w|)*(v-v_het)
      t_v = | 0 otherwise
        where
          v = w restricted to u (w where u is opaque)
    '''
    v = np.ma.masked_where(np.where(mcw[:,:,[2]] == 255, [1]*image_dim, [0]*image_dim)==0, image).filled(fill_value=0)
    v_het = get_image_het(v, image_dim, alpha_dim)
    t_v = get_image_t(v, image_dim, v_het, alpha_dim)
    '''
    9B. Draw Gaussian Sample,
      F = convolve(t_v, W)
      where
        W = Normalized Gaussian White Noise
    '''
    # TODO: Check if W is meant to be size of image, or a small window
    # TODO: I assume normalization here means by max and min (into the range -1 to 1) Or normalized as in, (value - value_het)/sqrt(value)?
    '''
    W = np.random.normal(0, 1, (3, 3, 1)).astype(np.float32)
    W = 2 * (W - np.min(W)) / (np.max(W) - np.min(W)) - 1
    F = np.zeros((t_v.shape[0], t_v.shape[1], 3))
    for x in range(t_v.shape[0]):
      for y in range(t_v.shape[1]):
        curr_F = [0, 0, 0]
        x_window_pos = 0
        for x_window in range(x-1, x+2):
          y_window_pos = 0
          for y_window in range(y-1, y+2):
            if (0 <= x_window < t_v.shape[0] and 0 <= y_window < t_v.shape[1] and t_v[x_window, y_window, 3] != 0):
              curr_F += W[x_window_pos, y_window_pos, 0] * t_v[x_window, y_window][:-1]
            y_window_pos += 1
          x_window_pos += 1
        F[x, y] = curr_F
    F = np.dstack((F, t_v[:,:,[3]]))'''
    
    W = np.random.normal(0, 1, (image_size[0], image_size[1], 1)).astype(np.float32)
    W = 2 * (W - np.min(W)) / (np.max(W) - np.min(W)) - 1
    F = convolve2d_fft(t_v, W)
    if alpha_dim:
      F[:,:,[alpha_dim]] = t_v[:,:,[alpha_dim]]
      F = np.where(F[:,:,[alpha_dim]] == 0, [0, 0, 0, 0], F)
    '''
    9C. Compute using CGD
      psi_1 = gamma_t |cxc (u|c - v_het)
      psi_2 = gamma_t |cxc (F|c)
    '''
    if alpha_dim:
      c = np.ma.masked_where(np.where(np.logical_and(mcw[:,:,[1]] == 255, v[:,:,[3]] > 0.001), [1]*image_dim, [0]*image_dim)==0, v).filled(fill_value=0) 
      constraint = np.where(c[:,:,[3]] == 0, [1]*image_dim, [0]*image_dim)
    else:
      c = np.ma.masked_where(np.where(mcw[:,:,[1]] == 255, [1]*image_dim, [0]*image_dim)==0, v).filled(fill_value=0)
      constraint = np.where(mcw[:,:,[1]] == 255, [1]*image_dim, [0]*image_dim)==0

    F_c = np.ma.masked_where(constraint, F).filled(fill_value=0)
    # TODO: Only works for grayscale, square images
    A = np.cov(F_c.reshape(F_c.shape[0], F_c.shape[1]))

    phi_1 = np.ma.masked_where(constraint, (v-v_het)).filled(fill_value=0)
    if alpha_dim:
      phi_1[:,:,[alpha_dim]] = c[:,:,[alpha_dim]]
    phi_1_shape = (phi_1.shape[0], phi_1.shape[1])
    psi_1 = []
    for dim in range(image_dim):
      psi_1_curr = CGD(phi_1[:,:,dim], A)
      #psi_1_curr = CGD(np.reshape(phi_1[:,:,[dim]], phi_1_shape), np.reshape(A[:,:,...], phi_1_shape))
      psi_1.append(psi_1_curr)
    psi_1 = np.dstack(tuple(psi_1))

    phi_2 = F_c
    if alpha_dim:
      phi_2[:,:,[alpha_dim]] = c[:,:,[alpha_dim]]
    phi_2_shape = (phi_2.shape[0], phi_2.shape[1])
    psi_2 = []
    for dim in range(image_dim):
      psi_2_curr = CGD(phi_1[:,:,dim], A)
      #psi_2_curr = CGD(np.reshape(phi_2[:,:,[dim]], phi_2_shape), np.reshape(A[:,:,...], phi_2_shape))
      psi_2.append(psi_2_curr)
    psi_2 = np.dstack(tuple(psi_2))
    '''
    9D. Extend psi_1 and psi_2 by zero-padding
    '''
    x_diff = max(0, (t_v.shape[0] - psi_1.shape[0]))
    x_left = int(x_diff / 2)
    x_right =  x_diff - x_left
    y_diff = max(0, (t_v.shape[1] - psi_1.shape[1]))
    y_top = int(y_diff / 2)
    y_bot = y_diff - y_top
    psi_1 = np.pad(psi_1, ((x_left, x_right), (y_top, y_bot), (0, 0)), 'constant')
    psi_2 = np.pad(psi_2, ((x_left, x_right), (y_top, y_bot), (0, 0)), 'constant')
    '''
    9E. Compute
      Kriging Component,
        (u - v_het)^* = convolve(convolve(t_v, t_v_tilde^T), psi_1)
      Innovation Component,
        F^* = convolve(convolve(t_v, t_v_tilde^T), psi_2)
        where
          convolve(t_v, t_v_tilde^T) = 1/|w| SUMx elem( wINTER(w-h) ) (u(x+h) - v_het)(u(x) - v_het)^T
    '''
    image_num = get_image_num(image, alpha_dim)
    cor_t_v = np.zeros((t_v.shape[0], t_v.shape[1], image_dim))
    for x in range(t_v.shape[0]):
      for y in range(t_v.shape[1]):
        curr_cor_t_v = [0] * image_dim
        curr_t_v = t_v[x, y]
        for x_window in range(x-1, x+2):
          for y_window in range(y-1, y+2):
            if (0 <= x_window < t_v.shape[0] and 0 <= y_window < t_v.shape[1]):
              if (alpha_dim):
                if (t_v[x_window, y_window, alpha_dim] != 0):
                  curr_cor_t_v += curr_t_v * t_v[x_window, y_window]
              else:
                curr_cor_t_v += curr_t_v * t_v[x_window, y_window]
        cor_t_v[x, y] = curr_cor_t_v
    if alpha_dim:
      cor_t_v[:,:,alpha_dim] = t_v[:,:,alpha_dim]
      cor_t_v = np.where(cor_t_v[:,:,[alpha_dim]] == 0, [0, 0, 0, 0], cor_t_v)

    print("CURR PSI SHAPE " + str(psi_1.shape))
    print("CURR CONV_T_V SHAPE " + str(cor_t_v.shape))
    kriging_comp = []
    for dim in range(image_dim):
      kriging_comp_curr = convolve2d_fft(cor_t_v[:,:,[dim]], psi_1[:,:,[dim]])
      kriging_comp.append(kriging_comp_curr)
    kriging_comp = np.dstack(tuple(kriging_comp))


    innov_comp = []
    for dim in range(image_dim):
      innov_comp_curr = convolve2d_fft(cor_t_v[:,:,[dim]], psi_2[:,:,[dim]])
      innov_comp.append(innov_comp_curr)
    innov_comp = np.dstack(tuple(innov_comp))
    '''
    9F. Fill M with values of v_het + (u - v_het)^* + F - F^*
    '''
    fill = v_het + kriging_comp + F - innov_comp
    result = np.zeros(image.shape)
    for x in range(mcw.shape[0]):
      for y in range(mcw.shape[1]):
        if (mcw[x, y, 0]) == 255:
          result[x, y] = fill[x, y]
        else:
          result[x, y] = image[x, y]
    debug(path_name, i, image_dim, v, t_v, F, cor_t_v, c, A, phi_1, phi_2, psi_1, psi_2, kriging_comp, innov_comp, result)
node.bypass(True)