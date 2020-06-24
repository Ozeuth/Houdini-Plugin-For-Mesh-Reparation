import argparse
import glob
import math
import numpy as np
import os
import sys
from itertools import combinations
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()

parser.add_argument("--image_path", type=str, default=os.path.dirname(os.path.abspath(__file__)) + "\\demo_inpaint")
parser.add_argument("--opening_suffix", type=str, default="opening")
parser.add_argument("--mask_suffix", type=str, default="mcw")

def convolve2d_fft(A, B):
  tfA = np.fft.fft2(A)
  tfB = np.fft.fft2(B)
  tfC = np.multiply(tfA, tfB)
  return np.real(np.fft.ifft2(tfC))

def lss(A, b):
  num_vars = A.shape[1]
  rank = np.linalg.matrix_rank(A)
  if rank == num_vars:     
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return (sol, True)
  else:
    sols = []
    for nz in combinations(range(num_vars), rank):
      try:
        sol = np.zeros((num_vars, 1))
        sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
        sols.append(sol)
      except np.linalg.LinAlgError:
        pass
    return (sols, False)

class Inpainter():
  def __init__(self, self_dict):
    for arg in self_dict: setattr(self, arg, self_dict[arg])
    self.image_paths = glob.glob(self.image_path + "/*" + self.opening_suffix + ".png")
    self.mcw_paths = glob.glob(self.image_path + "/*" + self.mask_suffix + ".png")
    self.threshold = 255 * 0.55

  def get_image_num(self, mask, image_dim):
    return int(np.sum(mask)/image_dim)

  def get_image_het(self, image, mask, image_dim):
    image_het = np.zeros(image_dim)
    for d in range(image_dim):
      image_het[d] = np.sum(np.ma.masked_where(mask[:, :, d] == 0, image[:, :, d]).filled(fill_value=0))
    image_het = image_het / self.get_image_num(mask, image_dim)
    return image_het 

  def inpaint_image(self, image, fill, mcw, alpha_dim):
    result = np.zeros(image.shape)
    for x in range(mcw.shape[0]):
      for y in range(mcw.shape[1]):
        if (mcw[x, y, 0] > self.threshold):
          result[x, y] = fill[x, y]
          if alpha_dim: result[x, y, alpha_dim] = 255
        else:
          result[x, y] = image[x, y]
    return result

  def debug(self, image_name, image_dim, v=None, F=None, F_result=None, cor_t_v=None, kriging_comp=None, innov_comp=None, full_result=None):
    if image_dim == 1:
      if v is not None: v = v.reshape((v.shape[0], v.shape[1]))
      if F is not None: F = F.reshape((F.shape[0], F.shape[1]))
      if F_result is not None: F_result = F_result.reshape((F_result.shape[0], F_result.shape[1]))
      if cor_t_v is not None: cor_t_v = cor_t_v.reshape((cor_t_v.shape[0], cor_t_v.shape[1]))
      if kriging_comp is not None: kriging_comp = kriging_comp.reshape((kriging_comp.shape[0], kriging_comp.shape[1]))
      if innov_comp is not None: innov_comp = innov_comp.reshape((innov_comp.shape[0], innov_comp.shape[1]))
      if full_result is not None: full_result = full_result.reshape((full_result.shape[0], full_result.shape[1]))
    if v is not None:
      im = Image.fromarray(np.uint8(v))
      im.save(self.image_path + "/" + str(image_name) + " v.png")
      im.close()
    if F is not None:
      im = Image.fromarray(np.uint8(F))
      im.save(self.image_path + "/" + str(image_name) + " F.png")
      im.close()
    if F_result is not None:
      im = Image.fromarray(np.uint8(F_result))
      im.save(self.image_path + "/" + str(image_name) + " F final.png")
      im.close()
    if cor_t_v is not None:    
      im = Image.fromarray(np.uint8(cor_t_v))
      im.save(self.image_path + "/" + str(image_name) + " cor_t_v.png")
      im.close()
    if kriging_comp is not None:
      im = Image.fromarray(np.uint8(kriging_comp))
      im.save(self.image_path + "/" + str(image_name) + " kriging.png")
      im.close()
    if innov_comp is not None:
      im = Image.fromarray(np.uint8(innov_comp))
      im.save(self.image_path + "/" + str(image_name) + " innov.png")
      im.close()
    if full_result is not None:
      im = Image.fromarray(np.uint8(full_result))
      im.save(self.image_path + "/" + str(image_name) + " final.png")
      im.close()

  def inpaint(self):
    '''
    9. 2D inpainting
    We follow B Galerne, A Leclaire[2017],
      inpainting using Gaussian conditional simulation, relying on a Kriging framework
      Special Thanks to:
        Gautier LOVEIKO for discussing his implementation
        Au Khai Xiang for providing mathematical insight
    '''
    for (image_path, mcw_path) in list(zip(self.image_paths, self.mcw_paths)):
      image_name = os.path.splitext(os.path.basename(image_path))[0]
      print("Starting inpainting of: " + str(image_name))
      v = F = F_result = cor_t_v = kriging_comp = innov_comp = full_result = alpha_dim = None
      image = Image.open(image_path).convert('LA')
      if (image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)):
        alpha_dim = len(image.split()) - 1
      image = np.array(image)
      image_dim = int(image.size / (image.shape[0] * image.shape[1]))
      image = image.reshape((image.shape[0], image.shape[1], image_dim))
      mcw = np.array(Image.open(mcw_path).convert('RGBA'))

      c_constraint = np.where(np.logical_and(mcw[:,:,[1]] > self.threshold, mcw[:,:,[3]] > self.threshold), [1]*image_dim, [0]*image_dim)
      w_constraint = np.where(np.logical_and(mcw[:,:,[2]] > self.threshold, mcw[:,:,[3]] > self.threshold), [1]*image_dim, [0]*image_dim)
      c_num = self.get_image_num(c_constraint, image_dim)
      w_num = self.get_image_num(w_constraint, image_dim)
      '''
      9A. Compute
        v_het = 1/|w|*[SUMr_elem(w)(v(r)), SUMg_elem(g)(v(g)), SUMb_elem(b)(v(b))]
              | 1/sqrt(|w|)*(v-v_het)
        t_v = | 0 otherwise
          where
            v = u restricted to w
      '''
      v = np.ma.masked_where(w_constraint==0, image).filled(fill_value=0)
      v_shape = (v.shape[0], v.shape[1], image_dim)

      v_het = self.get_image_het(v, w_constraint, image_dim)
      t_v = np.zeros(v_shape)
      t_v[:,:,...] = np.ma.masked_where(w_constraint==0, v-v_het).filled(fill_value=0) / math.sqrt(w_num)
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
      F_result = self.inpaint_image(image, F + v_het, mcw, alpha_dim)
      '''
      9C. Compute using LSS
        psi_1 = gamma_t |cxc (u|c - v_het)
        psi_2 = gamma_t |cxc (F|c)
      '''
      cor_t_v = np.zeros(v_shape)
      for dim in range(image_dim):
        cor_t_v[:,:,dim] = np.real(np.fft.ifft2(np.power(np.abs(np.fft.fft2(t_v[:,:,dim])),2)))

      u_cond = np.zeros((c_num, image_dim))
      F_cond = np.zeros((c_num, image_dim))
      mapping = np.zeros((c_num, 2))
      z = 0
      for x in range(mcw.shape[0]):
        for y in range(mcw.shape[1]):
          if mcw[x, y, 1] > self.threshold and mcw[x, y, 3] < self.threshold:
            u_cond[z] = v[x, y]
            F_cond[z] = F[x, y]
            mapping[z] = [x, y]
            z += 1

      gam_cond = np.zeros((c_num, c_num, image_dim))
      for elem_1 in range(c_num):
        for elem_2 in range(elem_1, c_num):
            position = mapping[elem_2] - mapping[elem_1]
            gam_cond[elem_1, elem_2] = cor_t_v[int(position[0]), int(position[1])]
      
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
            if mcw[x, y, 1] > self.threshold and mcw[x, y, 3] < self.threshold:
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
      full_result = self.inpaint_image(image, fill, mcw, alpha_dim)
      self.debug(image_name, image_dim, v, F, F_result, cor_t_v, kriging_comp, innov_comp, full_result)
      print("Finished inpainting of: " + str(image_name))

if __name__ == "__main__":
  # Run as is or start with optional arguments
  args = parser.parse_args()
  inpainter = Inpainter(vars(args))
  inpainter.inpaint()