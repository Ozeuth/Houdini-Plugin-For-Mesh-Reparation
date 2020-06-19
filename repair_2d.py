import glob
import math
import numpy as np
from itertools import combinations
import os
from PIL import Image, ImageDraw
def get_image_num(image, mask):
  return np.sum(mask)

def get_image_het(image, mask, image_dim):
  image_het = np.zeros(image_dim)
  for d in range(image_dim):
    image_het[d] = np.sum(np.ma.masked_where(mask == 0, image[:, :,[d]]).filled(fill_value=0))
  image_het = image_het / get_image_num(image, mask)
  return image_het

def get_image_t(image, mask, image_dim, image_het=None):
  t_image = np.zeros(image.shape)
  if image_het == None:
    image_het = get_image_het(image, mask, image_dim)
  t_image[:,:,...] = np.ma.masked_where(mask==0, image-image_het).filled(fill_value=0) / math.sqrt(get_image_num(image, mask))
  return t_image    
  
def get_image_cleaned(image, alpha, image_dim, alpha_dim=None):
  if alpha_dim:
    image[:,:,[alpha_dim]] = alpha[:,:,[alpha_dim]]
    image = np.where(image[:,:,[alpha_dim]] == 0, [0]*image_dim, image)
  return image

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

def diag(A):
  diagonal = np.zeros(A.shape)
  for x in range(A.shape[0]):
    for y in range(A.shape[1]):
      if x == y:
        diagonal[x, y] = A[x, y]
  return diagonal

def upper_tri(v):
  v_len = v.shape[0]
  vA = np.zeros((v_len * v_len, 1))
  for i in range(v_len):
    for i_ in range(i, (v_len-i)*v_len, v_len+1):
      vA[i_, 0] = v[i]
  A = vA.reshape((v_len, v_len))
  return A

def lower_tri(v):
  A = np.matrix(upper_tri(np.flip(v)))
  A = A.getH()
  A = np.array(A)
  B = A - diag(A)
  return B

def lss(A, b):
  num_vars = A.shape[1]
  rank = np.linalg.matrix_rank(A)
  if rank == num_vars:              
    sol = np.linalg.lstsq(A, b)[0] # not under-determined
    return (sol, True)
  else:
    sols = []
    for nz in combinations(range(num_vars), rank):# the variables not set to zero
      try: 
        sol = np.zeros((num_vars, 1))
        sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
        sols.append(sol)
      except np.linalg.LinAlgError:
        pass # picked bad variables, can't solve
    return (sols, False)

class Inpainter():
  def __init__(self, path_name, image_match="opening", mcw_match="mcw", alpha_dim=None):
    self.path_name = path_name
    self.alpha_dim = alpha_dim
    self.image_paths = glob.glob(path_name + "/*" + image_match + ".png")
    self.mcw_paths = glob.glob(path_name + "/*" + mcw_match + ".png")

  def debug(self, image_name, image_dim, v=None, t_v=None, F=None, cor_t_v=None, c=None, A=None, phi_1=None, phi_2=None, psi_1=None, psi_2=None, kriging_comp=None, innov_comp=None, result=None):
    #self.debug(image_name, image_dim, v, t_v, F, cor_t_v, c, A, phi_1, phi_2, psi_1, psi_2, kriging_comp, innov_comp, result)
    if image_dim == 1:
      if v is not None: v = v.reshape((v.shape[0], v.shape[1]))
      if t_v is not None: t_v = t_v.reshape((t_v.shape[0], t_v.shape[1]))
      if F is not None: F = F.reshape((F.shape[0], F.shape[1]))
      if cor_t_v is not None: cor_t_v = cor_t_v.reshape((cor_t_v.shape[0], cor_t_v.shape[1]))
      if c is not None: c = c.reshape((c.shape[0], c.shape[1]))
      if A is not None: A = A.reshape((A.shape[0], A.shape[1]))
      if phi_1 is not None: phi_1 = phi_1.reshape((phi_1.shape[0], phi_1.shape[1]))
      if phi_2 is not None: phi_2 = phi_2.reshape((phi_2.shape[0], phi_2.shape[1]))
      if psi_1 is not None: psi_1 = psi_1.reshape((psi_1.shape[0], psi_1.shape[1]))
      if psi_2 is not None: psi_2 = psi_2.reshape((psi_2.shape[0], psi_2.shape[1]))
      if kriging_comp is not None: kriging_comp = kriging_comp.reshape((kriging_comp.shape[0], kriging_comp.shape[1]))
      if innov_comp is not None: innov_comp = innov_comp.reshape((innov_comp.shape[0], innov_comp.shape[1]))
      if result is not None: result = result.reshape((result.shape[0], result.shape[1]))
    if v is not None:
      im = Image.fromarray(np.uint8(v))
      im.save(self.path_name + "/" + str(image_name) + " v.png")
      im.close()
    if t_v is not None:
      im = Image.fromarray(np.uint8(t_v))
      im.save(self.path_name + "/" + str(image_name) + " t_v.png")
      im.close()
    if F is not None:
      im = Image.fromarray(np.uint8(F))
      im.save(self.path_name + "/" + str(image_name) + " F.png")
      im.close()
    if cor_t_v is not None:    
      im = Image.fromarray(np.uint8(cor_t_v))
      im.save(self.path_name + "/" + str(image_name) + " cor_t_v.png")
      im.close()
    if c is not None:
      im = Image.fromarray(np.uint8(c))
      im.save(self.path_name + "/" + str(image_name) + " c.png")
      im.close()
    if A is not None:
      im = Image.fromarray(np.uint8(A))
      im.save(self.path_name + "/" + str(image_name) + " A.png")
      im.close()
    if phi_1 is not None:
      im = Image.fromarray(np.uint8(phi_1))
      im.save(self.path_name + "/" + str(image_name) + " phi_1.png")
      im.close()
    if phi_2 is not None:
      im = Image.fromarray(np.uint8(phi_2))
      im.save(self.path_name + "/" + str(image_name) + " phi_2.png")
      im.close()
    if psi_1 is not None:
      im = Image.fromarray(np.uint8(psi_1))
      im.save(self.path_name + "/" + str(image_name) +  " psi_1.png")
      im.close()
    if psi_2 is not None:
      im = Image.fromarray(np.uint8(psi_2))
      im.save(self.path_name + "/" + str(image_name) + " psi_2.png")
      im.close()
    if kriging_comp is not None:
      im = Image.fromarray(np.uint8(kriging_comp))
      im.save(self.path_name + "/" + str(image_name) + " kriging.png")
      im.close()
    if innov_comp is not None:
      im = Image.fromarray(np.uint8(innov_comp))
      im.save(self.path_name + "/" + str(image_name) + " innov.png")
      im.close()
    if result is not None:
      im = Image.fromarray(np.uint8(result))
      im.save(self.path_name + "/" + str(image_name) + " final.png")
      im.close()

  def inpaint(self):
    '''
    9. 2D inpainting
    We follow B Galerne, A Leclaire[2017],
      inpainting using Gaussian conditional simulation, relying on a Kriging framework
    '''
    for (image_path, mcw_path) in list(zip(self.image_paths, self.mcw_paths)):
      v = t_v = cor_t_v = c = A = phi_1 = phi_2 = psi_1 = psi_2 = kriging_comp = innov_comp = result = alpha_dim = None
      image = Image.open(image_path).convert('L')
      image_name = os.path.splitext(os.path.basename(image_path))[0]
      image = np.array(image)
      mcw = np.array(Image.open(mcw_path))
      alpha_dim = self.alpha_dim
      alpha_image = None
      image_dim = int(image.size / (image.shape[0] * image.shape[1]))
      image = image.reshape((image.shape[0], image.shape[1], image_dim))
      '''
      9A. Compute
        v_het = 1/|w|*[SUMr_elem(w)(v(r)), SUMg_elem(g)(v(g)), SUMb_elem(b)(v(b))]
              | 1/sqrt(|w|)*(v-v_het)
        t_v = | 0 otherwise
          where
            v = w restricted to u (w where u is opaque)
      '''
      w_constraint = np.where(mcw[:,:,[2]] == 255, [1]*image_dim, [0]*image_dim)
      v = np.ma.masked_where(w_constraint==0, image).filled(fill_value=0)
      if alpha_dim != None: alpha_image = v[:,:,[alpha_dim]]
      v_shape = (v.shape[0], v.shape[1], image_dim)

      v_het = get_image_het(v, w_constraint, image_dim)
      t_v = get_image_t(v, w_constraint, image_dim, v_het)
      t_v = get_image_cleaned(t_v, alpha_image, image_dim, alpha_dim)
      '''
      9B. Draw Gaussian Sample,
        F = convolve(t_v, W)
        where
          W = Normalized Gaussian White Noise
      '''
      W = np.random.normal(0, 1, (image.shape[0], image.shape[1], 1)).astype(np.float32)
      F = np.zeros(v_shape)
      for dim in range(image_dim):
        F[:,:,[dim]] = convolve2d_fft(t_v[:,:,[dim]], W)
      F = get_image_cleaned(F, alpha_image, image_dim, alpha_dim)
      '''
      9C. Compute using CGD
        psi_1 = gamma_t |cxc (u|c - v_het)
        psi_2 = gamma_t |cxc (F|c)
      '''
      c_constraint = np.where(mcw[:,:,[1]] == 255, [1]*image_dim, [0]*image_dim)
      t_v_num = get_image_num(t_v, c_constraint)
      cor_t_v = np.zeros(v_shape)
      for dim in range(image_dim):
        cor_t_v[:,:,[dim]] = np.real(np.fft.ifft2(np.power(np.abs(np.fft.fft2(t_v[:,:,[dim]])),2)))

      min_c_x = mcw.shape[0]
      min_c_y = mcw.shape[1]
      max_c_x = 0
      max_c_y = 0
      for x in range(mcw.shape[0]):
        for y in range(mcw.shape[1]):
          if (mcw[x, y, 1] == 255):
            min_c_x = min(min_c_x, x)
            min_c_y = min(min_c_y, y)
            max_c_x = max(max_c_x, x)
            max_c_y = max(max_c_y, y)

      v_1 = []
      v_2 = []
      for dim in range(image_dim):
        v_1.append(np.zeros((1, t_v_num)))
        v_2.append(np.zeros((1, t_v_num)))
      v_1 = np.dstack(tuple(v_1))
      v_2 = np.dstack(tuple(v_2))
      z = 0
      for x in range(min_c_x, max_c_x + 1):
        for y in range(min_c_y, max_c_y + 1):
          if (mcw[x, y, 1] == 255):
            for dim in range(image_dim):
              v_1[0, z, dim] = cor_t_v[x - min_c_x, y - min_c_y, dim]
              if (y - max_c_y == 0):
                v_2[0, z, dim] = cor_t_v[x - min_c_x, 0, dim]
              else:  
                v_2[0, z, dim] = cor_t_v[x - min_c_x, y - max_c_y + t_v.shape[1], dim]
            z += 1

      gam_cond = []
      for dim in range(image_dim):
        gam_cond.append(np.zeros((t_v_num, t_v_num, 2)))
      gam_cond = np.array(gam_cond)

      gam_cond = []
      for dim in range(image_dim):
        U_curr = upper_tri(v_1[0,:,dim])
        L_curr = lower_tri(v_2[0,:,dim])
        gam_cond_curr = U_curr + L_curr
        gam_cond.append(gam_cond_curr)
      gam_cond = np.dstack(tuple(gam_cond))

      for dim in range(image_dim):
        gam_cond[:,:,dim] = np.transpose(gam_cond[:,:,dim]) + gam_cond[:,:,dim] - diag(gam_cond[:,:,dim])
      
      '''
      z = 0
      for x in range(min_c_x, max_c_x):
        for y in range(min_c_y, max_c_y):
          if (mcw[x, y, 1] == 255):
            # valid fixed point
            z_ = 0
            for x_ in range(min_c_x, x):
              for y_ in range(min_c_y, y):
                if (mcw[x_, y_, 1] == 255):
                  # valid compared point
                  for dim in range(image_dim):
                    gam_cond[z, z_, dim] = [0, 0]
                  z_ += 1
            z_ = 0
            for x_ in range(min_c_x, max_c_x):
              for y_ in range(min_c_y, max_c_y):
                if (mcw[x_, y_, 1] == 255):
                  # valid compared point
                  z_diff = z_ - z
                  for dim in range(image_dim):
                    M_temp1_curr = upper_tri(v_1[_,_,dim])
                    gam_cond[_,_,dim] = M_temp1_curr
                    if (x_ > x): # Tentative
                      M_temp2_curr = lower_tri(v_2[_,_,dim])
                      gam_cond[_,_,dim] += M_temp2_curr
                  z_ += 1
            z += 1

      for dim in range(image_dim):
        gam_cond[:,:,dim] = np.transpose(gam_cond[:,:,dim]) + gam_cond[:,:,dim] - diag(gam_cond[:,:,dim])'''
      
      u_cond = []
      F_cond = []
      for dim in range(image_dim):
        #u_cond.append(np.zeros((1, t_v_num)))
        #F_cond.append(np.zeros((1, t_v_num)))
        u_cond.append(np.zeros(t_v_num))
        F_cond.append(np.zeros(t_v_num))
      u_cond = np.dstack(tuple(u_cond))
      F_cond = np.dstack(tuple(F_cond))
      z = 0
      for x in range(min_c_x, max_c_x):
        for y in range(min_c_y, max_c_y):
          if mcw[x, y, 1] == 255:
            for dim in range(image_dim):
              u_cond[0, z, dim] = v[x, y, dim]
              F_cond[0, z, dim] = F[x, y, dim]
            z += 1
      
      psi_1 = []
      psi_2 = []
      for dim in range(image_dim):
        sol, is_singular = lss(gam_cond[:,:,dim], np.transpose(u_cond[:,:,dim] - v_het[dim]))
        if is_singular:
          psi_1_curr = sol
        sol, is_singular = lss(gam_cond[:,:,dim], np.transpose(F_cond[:,:,dim]))
        if is_singular:
          psi_2_curr = sol
        psi_1.append(psi_1_curr)
        psi_2.append(psi_2_curr)
      psi_1 = np.dstack(tuple(psi_1))
      psi_2 = np.dstack(tuple(psi_2))
      '''
      9D. Extend psi_1 and psi_2 by zero-padding
      '''
      '''
      x_diff = max(0, (t_v.shape[0] - psi_1.shape[0]))
      x_left = int(x_diff / 2)
      x_right =  x_diff - x_left
      y_diff = max(0, (t_v.shape[1] - psi_1.shape[1]))
      y_top = int(y_diff / 2)
      y_bot = y_diff - y_top
      psi_1 = np.pad(psi_1, ((x_left, x_right), (y_top, y_bot), (0, 0)), 'constant')
      psi_2 = np.pad(psi_2, ((x_left, x_right), (y_top, y_bot), (0, 0)), 'constant')'''
      '''
      9E. Compute
        Kriging Component,
          (u - v_het)^* = convolve(convolve(t_v, t_v_tilde^T), psi_1)
        Innovation Component,
          F^* = convolve(convolve(t_v, t_v_tilde^T), psi_2)
          where
            convolve(t_v, t_v_tilde^T) = 1/|w| SUMx elem( wINTER(w-h) ) (u(x+h) - v_het)(u(x) - v_het)^T
      '''
      kriging_comp = []
      for dim in range(image_dim):
        kriging_comp_curr = convolve2d_fft(cor_t_v[:,:,[dim]], psi_1[:,:,[dim]])
        kriging_comp.append(kriging_comp_curr)
      kriging_comp = np.dstack(tuple(kriging_comp))
      
      innov_comp = []
      for dim in range(image_dim):
        innov_comp_curr = F[:,:,[dim]] - convolve2d_fft(cor_t_v[:,:,[dim]], psi_2[:,:,[dim]])
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
      self.debug(image_name, image_dim, v, t_v, F, cor_t_v, c, A, phi_1, phi_2, psi_1, psi_2, kriging_comp, innov_comp, result)

if __name__ == "__main__":
  inpainter = Inpainter("C:\\Users\\Ozeuth\\Python-Houdini-Mesh-Repair\\demo_inpaint")
  inpainter.inpaint()