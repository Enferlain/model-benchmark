import torch
import math


class SlidingSNRLossAwareSampler:
    """
    Sliding window loss-aware sampler for diffusion training.

    - Curriculum: window center moves (in log-SNR space) as training progresses,
      focusing sampling where refinement or denoising is most relevant to training progress.
    - Loss-aware: bins with higher EMA (Exponential Moving Average) loss are sampled more often;
      this prioritizes regions where the model struggles, boosting learning efficiency.
    - Compatible API and device handling for integration with prior samplers.
    """

    def __init__(
            self,
            noise_scheduler,
            num_bins: int = 64,
            ema_beta: float = 0.90,                     # EMA timescale for bin loss tracking
            temperature: float = 0.35,                  # Controls sharpness of loss-prior distribution
            min_prob: float = 1e-3,                     # Prevents probability starvation of any bin
            entropy_floor_ratio: float = 0.8,           # Triggers mixing-in of uniform if distribution collapses
            uniform_mix_when_low_entropy: float = 0.1,  # How much uniform to mix if low entropy
            uniform_tail_prob: float = 0.05,            # Ensures all bins have some mass ("probability tail")
            total_training_steps: int = 1000,           # Defines training curriculum duration
            start_center_mu: float = None,              # Initial window center (log-SNR)
            end_center_mu: float = None,                # Final window center (log-SNR)
            start_half_width: float = 0.5,              # Initial window half-width
            end_half_width: float = 1.2,                # Final window half-width
            cap_max_t: int = None,
            cap_target_t: int = None,
            cap_ema_beta: float = 0.9,
            cap_saturation_thresh: float = 0.25,
            cap_step_min: int = 5,
            cap_step_max_frac: float = 0.1,
            device=torch.device('cpu'),
    ):
        self.device = device

        # --- SNR statistics, mapping and curriculum initialization ---
        a2 = noise_scheduler.alphas_cumprod.float().clamp(1e-12, 1.0 - 1e-12)
        snr = a2 / (1.0 - a2)
        log_snr = torch.log(snr.clamp(min=1e-20)).to(self.device)

        self.log_snr_min = float(log_snr.min())
        self.log_snr_max = float(log_snr.max())
        self.log_snr_median = float(log_snr.median())
        self.log_snr_80th = float(log_snr.quantile(0.8))

        # Use recommended starting values if not provided
        # These are calibrated for typical DDPM/SDXL schedules
        self.start_center_mu = self.log_snr_median if start_center_mu is None else start_center_mu
        self.end_center_mu = self.log_snr_80th if end_center_mu is None else end_center_mu
        self.start_half_width = start_half_width
        self.end_half_width = end_half_width

        self.num_bins = num_bins
        self.ema_beta = ema_beta
        self.temperature = temperature
        self.min_prob = min_prob
        self.entropy_floor_ratio = entropy_floor_ratio
        self.uniform_mix_when_low_entropy = uniform_mix_when_low_entropy
        self.uniform_tail_prob = uniform_tail_prob
        self.total_training_steps = total_training_steps

        # --- Cap/tail parameters ---
        self.cap_max_t = cap_max_t
        self.cap_target_t = cap_target_t if cap_target_t is not None else noise_scheduler.num_train_timesteps - 1
        self.cap_ema_beta = cap_ema_beta
        self.cap_saturation_thresh = cap_saturation_thresh
        self.cap_step_min = cap_step_min
        self.cap_step_max_frac = cap_step_max_frac

        self.global_step = 0
        self.num_train_timesteps = noise_scheduler.num_train_timesteps

        # --- Log-SNR and bin mappings on device ---
        self.log_snr_original = log_snr.contiguous()
        self.log_snr_sorted, self.sort_indices = torch.sort(self.log_snr_original)
        self.inverse_sort_indices = torch.argsort(self.sort_indices)
        max_bin = min(num_bins - 1, len(self.log_snr_sorted) - 1)

        # Assign each timestep (index) to a SNR bin, sorted in log-SNR
        self.timestep_to_bin = torch.searchsorted(
            self.log_snr_sorted, self.log_snr_original, right=True
        ).clamp(max=max_bin)
        self.num_actual_bins = int(self.timestep_to_bin.max().item() + 1)

        # --- Precompute bin-to-timestep-list for fast index sampling ---
        self.bin_to_timesteps = [
            (self.timestep_to_bin == b).nonzero(as_tuple=True)[0] for b in range(self.num_actual_bins)
        ]
        self.t_indices = torch.arange(self.num_train_timesteps, device=device)

        # --- Buffers ---
        self.ema_loss = torch.zeros(self.num_actual_bins, device=device)
        self.ema_prob = torch.ones(self.num_actual_bins, device=device) / self.num_actual_bins
        self.ema_prob_cap_boundary = torch.tensor(0.0, device=device)
        self.cap_max_t_current = self.cap_max_t if self.cap_max_t is not None else self.cap_target_t

    def _interpolate(self, start, end, frac):
        # Linear interpolation helper for sliding window
        return start + frac * (end - start)

    def _get_current_window(self):
        # Curriculum window: window center and width slide from start to end values
        progress = min(self.global_step / max(self.total_training_steps, 1), 1.0)
        center = self._interpolate(self.start_center_mu, self.end_center_mu, progress)
        half_width = self._interpolate(self.start_half_width, self.end_half_width, progress)
        return center, half_width

    def _gaussian_weights_from_center(self, center, half_width):
        # Gaussian prior: highlights central window, downweights far bins
        dist = torch.abs(self.log_snr_sorted - center)
        weights = torch.exp(-0.5 * (dist / half_width) ** 2)
        return weights / weights.sum()

    def update(self, timesteps, loss):
        """
        Update loss statistics and adjust sampling probabilities.
        - `timesteps`: tensor of sampled timesteps from training batch
        - `loss`: tensor of per-sample training objective for those timesteps
        """
        bins = self.timestep_to_bin[timesteps.to(self.device)]

        # Update EMA buffer with batch loss for sampled bins
        for i in range(len(loss)):
            b = int(bins[i].item())
            l = float(loss[i].item())
            self.ema_loss[b] = self.ema_beta * self.ema_loss[b] + (1 - self.ema_beta) * l

        # Loss-aware raw probabilities (higher loss = more samples)
        raw_probs = torch.clamp(self.ema_loss, min=1e-12)
        raw_probs = raw_probs / raw_probs.sum() if raw_probs.sum() > 0 else torch.ones_like(raw_probs) / len(raw_probs)
        tempered_probs = raw_probs.pow(1.0 / self.temperature)
        tempered_probs = tempered_probs / tempered_probs.sum()

        # Sliding window: bias with curriculum window (Gaussian prior)
        center, half_width = self._get_current_window()
        window_weights = self._gaussian_weights_from_center(center, half_width)
        combined_probs = tempered_probs * window_weights
        combined_probs = combined_probs / combined_probs.sum()

        # Uniform probability tail to avoid zero-prob bins (prevents collapse)
        combined_probs = torch.clamp(combined_probs, min=self.min_prob)
        combined_probs = combined_probs / combined_probs.sum()
        uniform = torch.ones_like(combined_probs) / len(combined_probs)
        combined_probs = (1.0 - self.uniform_tail_prob) * combined_probs + self.uniform_tail_prob * uniform
        combined_probs = combined_probs / combined_probs.sum()

        # Entropy guard: mix uniform if distribution is too sharp
        entropy = -torch.sum(combined_probs * torch.log(combined_probs + 1e-12))
        max_ent = math.log(len(combined_probs))
        if entropy / max_ent < self.entropy_floor_ratio:
            combined_probs = (
                                         1.0 - self.uniform_mix_when_low_entropy) * combined_probs + self.uniform_mix_when_low_entropy * uniform
            combined_probs = combined_probs / combined_probs.sum()

        self.ema_prob = combined_probs.clone()
        self.global_step += 1

    def sample(self, batch_size, device=None, global_step=None, max_steps=None, **_):
        """
        Draw batch of timesteps for training, biasing by curriculum and loss-aware difficulty.
        - Outputs tensor on requested device, compatible with training data pipeline.
        """
        if global_step is not None:
            self.global_step = global_step
        probs = self.ema_prob.to(self.device)

        # Sample bins (loss-prior and curriculum-window weighted)
        bin_indices = torch.multinomial(probs, batch_size, replacement=True)
        timesteps = torch.empty(batch_size, dtype=torch.long)
        for i, b in enumerate(bin_indices.tolist()):
            t_idxs = self.bin_to_timesteps[b]
            # Uniform sample from timesteps in this bin
            if len(t_idxs) > 0:
                t_idx = t_idxs[torch.randint(len(t_idxs), (1,))]
                timesteps[i] = t_idx
            else:
                timesteps[i] = torch.randint(self.num_train_timesteps, (1,))
        return timesteps.to(device if device is not None else self.device)
