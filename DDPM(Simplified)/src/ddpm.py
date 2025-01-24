import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDPMSampler:

    def __init__(
            self,
            generator:torch.Generator,
            num_training_steps = 1000,
            beta_start:float = 0.00085,
            beta_end:float = 0.0120,
            ):
          self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps,dtype=torch.float32) ** 2
          self.alphas = 1.0 - self.betas
          self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)#[a_0,a_0*a_1,a_0*a_1*a_2,...]
          self.one  = torch.tensor(1.0)

          self.generator = generator
          self.num_training_steps = num_training_steps
          self.timesteps =  torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps):
         self.num_inference_steps = num_inference_steps

         #999,998,997,...,1,0 -> 999,979,959,...,0 range(999,0,-20)
         step_ratio = self.num_training_steps // self.num_inference_steps
         timesteps = self.timesteps.copy()[::step_ratio].round()[::-1].astype(np.int64)
         self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self,t:int)->int:
         prev_t = t - (self.num_training_steps // self.num_inference_steps)
         return prev_t

    def _get_variance(self,t:int)->torch.Tensor:
         prev_t = self._get_previous_timestep(t)
         alpha_prod_t = self.alphas_cumprod[t]
         alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
         current_beta_t = 1 - alpha_prod_t / alpha_prod_prev_t #beta_t = 1 - alpha_t = 1- alpha_t_prod / alpha_t_prod_prev
         #Eq.(7) of DDPM paper
         variance = (1-alpha_prod_prev_t)/(1-alpha_prod_t)*current_beta_t
         variance = torch.clamp(variance,min = 1e-20)
         return variance
    
    
    def set_strength(self, strength = 1):
         start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
         self.timesteps = self.timesteps[start_step:]
         self.start_step = start_step


    def step(self,timestep:int,latents:torch.Tensor,model_output:torch.Tensor):
         t = timestep
         prev_t = self._get_previous_timestep(t)

         alpha_prod_t = self.alphas_cumprod[t]
         alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
         beta_prod_t  = 1 - alpha_prod_t
         beta_prod_prev_t = 1 - alpha_prod_prev_t
         current_alpha_t  = alpha_prod_t / alpha_prod_prev_t
         current_beta_t  = 1 - current_alpha_t

         #Eq.(15) of DDPM paper
         pred_original_sample = (latents - (beta_prod_t ** 0.5) * model_output) / alpha_prod_t ** 0.5

         #Eq.(7) of DDPM paper
         pred_original_sample_coeff = (alpha_prod_prev_t ** 0.5 * current_beta_t) / beta_prod_t
         current_sample_coeff  = current_alpha_t ** 0.5 * beta_prod_prev_t / beta_prod_t
         pred_current_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

         variance = 0
         if t > 0:
              device = model_output.device
              noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
              variance = self._get_variance(t) ** 0.5 * noise
         pred_prev_sample = pred_current_sample + variance
         return pred_prev_sample

    def add_noise(self, original_samples:torch.FloatTensor, timesteps:torch.IntTensor)->torch.FloatTensor:
         alpha_cumprod = self.alphas_cumprod.to(device = original_samples.device,dtype=original_samples.dtype)
         timesteps = timesteps.to(device = original_samples.device)

         sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[timesteps])
         sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
         while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
              sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)

         sqrt_one_minus_alpha_cumprod = torch.sqrt((1.0 - alpha_cumprod[timesteps])) #std
         sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()
         while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
              sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

         #Eq.(4) of DDPM paper
         #Z = N(0,1) -> N(mean,variance) = X
         #X = Z * std + mean
         noise = torch.randn(
              original_samples.shape, 
              generator=self.generator, 
              device=original_samples.device,
              dtype=original_samples.dtype
              )
         noisy_samples = (sqrt_alpha_cumprod * original_samples) + (sqrt_one_minus_alpha_cumprod * noise)
         return noisy_samples



         
