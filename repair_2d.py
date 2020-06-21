import glob
import math
import numpy as np
from itertools import combinations
import os
from PIL import Image, ImageDraw

def get_image_num(mask):
  return np.sum(mask)

def get_image_het(image, mask, image_dim):
  image_het = np.zeros(image_dim)
  for d in range(image_dim):
    image_het[d] = np.sum(np.ma.masked_where(mask == 0, image[:, :,[d]]).filled(fill_value=0))
  image_het = image_het / get_image_num(mask)
  return image_het 
  
def get_image_cleaned(image, alpha, image_dim, alpha_dim=None):
  if alpha_dim:
    image[:,:,[alpha_dim]] = alpha[:,:,[alpha_dim]]
    image = np.where(image[:,:,[alpha_dim]] == 0, [0]*image_dim, image)
  return image

def convolve2d_fft(A, B):
  tfA = np.fft.fft2(A)
  tfB = np.fft.fft2(B)
  tfC = np.multiply(tfA, tfB)
  return np.real(np.fft.ifft2(tfC))

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
  '''
  A = np.matrix(A)
  A = A.getH()
  A = np.array(A)'''
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


  def inpaint_image(self, image, fill, mcw):
    result = np.zeros(image.shape)
    for x in range(mcw.shape[0]):
      for y in range(mcw.shape[1]):
        if (mcw[x, y, 0] > 255/2):
          result[x, y] = fill[x, y]
        else:
          result[x, y] = image[x, y]
    return result

  def debug(self, image_name, image_dim, v=None, F=None, F_result=None, cor_t_v=None, c=None, A=None, psi_1=None, psi_2=None, kriging_comp=None, innov_comp=None, full_result=None):
    if image_dim == 1:
      if v is not None: v = v.reshape((v.shape[0], v.shape[1]))
      if F is not None: F = F.reshape((F.shape[0], F.shape[1]))
      if F_result is not None: F_result = F_result.reshape((F_result.shape[0], F_result.shape[1]))
      if cor_t_v is not None: cor_t_v = cor_t_v.reshape((cor_t_v.shape[0], cor_t_v.shape[1]))
      if c is not None: c = c.reshape((c.shape[0], c.shape[1]))
      if A is not None: A = A.reshape((A.shape[0], A.shape[1]))
      if psi_1 is not None: psi_1 = psi_1.reshape((psi_1.shape[0], psi_1.shape[1]))
      if psi_2 is not None: psi_2 = psi_2.reshape((psi_2.shape[0], psi_2.shape[1]))
      if kriging_comp is not None: kriging_comp = kriging_comp.reshape((kriging_comp.shape[0], kriging_comp.shape[1]))
      if innov_comp is not None: innov_comp = innov_comp.reshape((innov_comp.shape[0], innov_comp.shape[1]))
      if full_result is not None: full_result = full_result.reshape((full_result.shape[0], full_result.shape[1]))
    if v is not None:
      im = Image.fromarray(np.uint8(v))
      im.save(self.path_name + "/" + str(image_name) + " v.png")
      im.close()
    if F is not None:
      im = Image.fromarray(np.uint8(F))
      im.save(self.path_name + "/" + str(image_name) + " F.png")
      im.close()
    if F_result is not None:
      im = Image.fromarray(np.uint8(F_result))
      im.save(self.path_name + "/" + str(image_name) + " F final.png")
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
    if full_result is not None:
      im = Image.fromarray(np.uint8(full_result))
      im.save(self.path_name + "/" + str(image_name) + " final.png")
      im.close()

  def inpaint(self):
    '''
    9. 2D inpainting
    We follow B Galerne, A Leclaire[2017],
      inpainting using Gaussian conditional simulation, relying on a Kriging framework
    '''
    for (image_path, mcw_path) in list(zip(self.image_paths, self.mcw_paths)):
      v = F = F_result = cor_t_v = c = A = psi_1 = psi_2 = kriging_comp = innov_comp = full_result = alpha_dim = None
      image = Image.open(image_path).convert('L')
      image_name = os.path.splitext(os.path.basename(image_path))[0]
      image = np.array(image)
      mcw = np.array(Image.open(mcw_path))
      alpha_dim = self.alpha_dim
      alpha_image = None
      image_dim = int(image.size / (image.shape[0] * image.shape[1]))
      image = image.reshape((image.shape[0], image.shape[1], image_dim))

      threshold = 255 * 0.55
      m_constraint = np.where(np.logical_and(mcw[:,:,[0]] > threshold, mcw[:,:,[alpha_dim]] > threshold if alpha_dim != None else True), [1]*image_dim, [0]*image_dim)
      c_constraint = np.where(np.logical_and(mcw[:,:,[1]] > threshold, mcw[:,:,[alpha_dim]] > threshold if alpha_dim != None else True), [1]*image_dim, [0]*image_dim)
      w_constraint = np.where(np.logical_and(mcw[:,:,[2]] > threshold, mcw[:,:,[alpha_dim]] > threshold if alpha_dim != None else True), [1]*image_dim, [0]*image_dim)
      m_num = get_image_num(m_constraint)
      c_num = get_image_num(c_constraint)
      w_num = get_image_num(w_constraint)
      '''
      9A. Compute
        v_het = 1/|w|*[SUMr_elem(w)(v(r)), SUMg_elem(g)(v(g)), SUMb_elem(b)(v(b))]
              | 1/sqrt(|w|)*(v-v_het)
        t_v = | 0 otherwise
          where
            v = u restricted to w
      '''
      v = np.ma.masked_where(w_constraint==0, image).filled(fill_value=0)
      if alpha_dim != None: alpha_image = v[:,:,[alpha_dim]]
      v_shape = (v.shape[0], v.shape[1], image_dim)

      v_het = get_image_het(v, w_constraint, image_dim)
      t_v = np.zeros(v_shape)
      t_v[:,:,...] = np.ma.masked_where(w_constraint==0, v-v_het).filled(fill_value=0) / math.sqrt(w_num)
      t_v = get_image_cleaned(t_v, alpha_image, image_dim, alpha_dim)
      '''
      9B. Draw Gaussian Sample,
        F = convolve(t_v, W)
        where
          W = Normalized Gaussian White Noise
      '''
      W = np.random.normal(0, 1, (image.shape[0], image.shape[1])).astype(np.float32)
      F = np.zeros(v_shape)
      for dim in range(image_dim):
        F[:,:,dim] = convolve2d_fft(t_v[:,:,dim], W)
      F = get_image_cleaned(F, alpha_image, image_dim, alpha_dim)
      F_result = self.inpaint_image(image, F + v_het, mcw)
      '''
      9C. Compute using CGD
        psi_1 = gamma_t |cxc (u|c - v_het)
        psi_2 = gamma_t |cxc (F|c)
      '''
      cor_t_v = np.zeros(v_shape)
      for dim in range(image_dim):
        cor_t_v[:,:,dim] = np.real(np.fft.ifft2(np.power(np.abs(np.fft.fft2(t_v[:,:,dim])),2)))

      min_c_x = mcw.shape[0]
      min_c_y = mcw.shape[1]
      max_c_x = 0
      max_c_y = 0
      for x in range(mcw.shape[0]):
        for y in range(mcw.shape[1]):
          if (mcw[x, y, 1] > threshold):
            min_c_x = min(min_c_x, x)
            min_c_y = min(min_c_y, y)
            max_c_x = max(max_c_x, x)
            max_c_y = max(max_c_y, y)

      v_1 = np.zeros((c_num, image_dim))
      v_2 = np.zeros((c_num, image_dim))
      z = 0
      for x in range(min_c_x, max_c_x + 1):
        for y in range(min_c_y, max_c_y + 1):
          if (mcw[x, y, 1] > threshold):
            for dim in range(image_dim):
              v_1[z, dim] = cor_t_v[x - min_c_x, y - min_c_y, dim]
              if (y - max_c_y == 0):
                v_2[z, dim] = cor_t_v[x - min_c_x, 0, dim]
              else:  
                v_2[z, dim] = cor_t_v[x - min_c_x, y - max_c_y + t_v.shape[1], dim]
            z += 1

      u_cond = np.zeros((c_num, image_dim))
      F_cond = np.zeros((c_num, image_dim))

      cor_t_v_cond = np.zeros((c_num, image_dim))
      t_v_cond = np.zeros((c_num, image_dim))
      z = 0
      for x in range(min_c_x, max_c_x + 1):
        for y in range(min_c_y, max_c_y + 1):
          if mcw[x, y, 1] > threshold:
            for dim in range(image_dim):
              u_cond[z, dim] = v[x, y, dim]
              F_cond[z, dim] = F[x, y, dim]

              cor_t_v_cond[z, dim] = cor_t_v[x, y, dim]
              t_v_cond[z, dim] = t_v[x, y, dim]
            z += 1
      '''
      gam_cond = np.zeros((c_num, c_num, image_dim))
      z_prior = 0
      for x in range(min_c_x, max_c_x + 1):
        z_curr = z_prior
        for y in range(min_c_y, max_c_y + 1):
          if (mcw[x, y, 1] > threshold):
            z_curr += 1

        z_prior_ = z_curr
        for x_ in range(min_c_x, max_c_x + 1):
          z_curr_ = z_prior_
          for y_ in range(min_c_y, max_c_y + 1):
            if (mcw[x_, y_, 1] > threshold):
              z_curr_ += 1
            
          z_ij_ = z_prior_ - z_prior
          z_ij = z_curr_ - z_curr
          #print(str(z_ij_) + " " + str(z_ij) + " " + str(z_prior_) + " " + str(z_prior) + " " + str(z_curr_) + " " + str(z_curr)) 
          for dim in range(image_dim):
            bin_1_curr = c_cond[z_prior:z_curr]
            bin_2_curr = c_cond[z_prior_:z_curr_]
            print(bin_1_curr)
            print(bin_2_curr)
            print(bin_1_curr.shape)
            print(bin_2_curr.shape)
            M_temp_1_curr= upper_tri(v_1[z_ij_: z_ij + 1,dim])
            print("M1")
            print(M_temp_1_curr)
            print(M_temp_1_curr.shape)
            print(gam_cond[z_prior:z_curr, z_prior_:z_curr_, dim].shape)
            gam_cond[z_prior:z_curr, z_prior_:z_curr_, dim] += M_temp_1_curr
            if z_curr_ > z_curr:
              M_temp_2_curr = lower_tri(v_2[z_ij_: z_ij + 1,dim])
              print("M2")
              print(M_temp_2_curr)
              gam_cond[z_prior:z_curr, z_prior_:z_curr_, dim] += M_temp_2_curr
          z_prior_ = z_curr_
        z_prior = z_curr'''

      U = []
      L = []
      for dim in range(image_dim):
        U_curr = upper_tri(v_1[:,dim])
        L_curr = lower_tri(v_2[:,dim])
        U.append(U_curr)
        L.append(L_curr)
      U = np.dstack(tuple(U))
      L = np.dstack(tuple(L))

      gam_cond = np.zeros((c_num, c_num, image_dim))
      for elem_1 in range(U.shape[0]):
        for elem_2 in range(U.shape[0]):
          for dim in range(image_dim):
            if (elem_2 > elem_1):
              gam_cond[elem_1, elem_2, dim] = U[elem_1, elem_2, dim] + L[elem_1, elem_2, dim]
            else:
              gam_cond[elem_1, elem_2, dim] = U[elem_1, elem_2, dim]
      for dim in range(image_dim):
        gam_cond[:,:,dim] = np.transpose(gam_cond[:,:,dim]) + gam_cond[:,:,dim] - diag(gam_cond[:,:,dim])
      
      psi_1 = np.zeros((c_num, image_dim))
      psi_2 = np.zeros((c_num, image_dim))
      for dim in range(image_dim):
        sol, is_singular = lss(gam_cond[:,:,dim], np.transpose(u_cond[:,dim] - v_het[dim]))
        if is_singular:
          psi_1[:,dim] = sol
        sol, is_singular = lss(gam_cond[:,:,dim], np.transpose(F_cond[:,dim]))
        if is_singular:
          psi_2[:,dim]= sol
      '''
      9D. Extend psi_1 and psi_2 by zero-padding
      '''
      psi_1_ = np.zeros((t_v.shape[0], t_v.shape[1], image_dim))
      psi_2_ = np.zeros((t_v.shape[0], t_v.shape[1], image_dim))
      for dim in range(image_dim):
        z = 0
        for x in range(mcw.shape[0]):
          for y in range(mcw.shape[1]):
            if mcw[x, y, 1] > threshold:
              psi_1_[x, y, dim] = psi_1[z,dim]
              psi_2_[x, y, dim] = psi_2[z,dim]
              z += 1
      psi_1 = psi_1_
      psi_2 = psi_2_
      '''
      9E. Compute
        Kriging Component,
          (u - v_het)^* = convolve(convolve(t_v, t_v_tilde^T), psi_1)
        Innovation Component,
          F^* = convolve(convolve(t_v, t_v_tilde^T), psi_2)
          where
            convolve(t_v, t_v_tilde^T) = 1/|w| SUMx elem( wINTER(w-h) ) (u(x+h) - v_het)(u(x) - v_het)^T
      '''
      kriging_comp = np.zeros((t_v.shape[0], t_v.shape[1], image_dim))
      for dim in range(image_dim):
        kriging_comp[:,:,dim] = convolve2d_fft(cor_t_v[:,:,dim], psi_1[:,:,dim])
      
      innov_comp = np.zeros((t_v.shape[0], t_v.shape[1], image_dim))
      for dim in range(image_dim):
        innov_comp[:,:,dim] = convolve2d_fft(cor_t_v[:,:,dim], psi_2[:,:,dim])
      '''
      9F. Fill M with values of v_het + (u - v_het)^* + F - F^*
      '''
      fill =  kriging_comp +  F - innov_comp + v_het
      full_result = self.inpaint_image(image, fill, mcw)
      self.debug(image_name, image_dim, v, F, F_result, cor_t_v, c, A, psi_1, psi_2, kriging_comp, innov_comp, full_result)

if __name__ == "__main__":
  inpainter = Inpainter("C:\\Users\\Ozeuth\\Python-Houdini-Mesh-Repair\\demo_inpaint")
  inpainter.inpaint()