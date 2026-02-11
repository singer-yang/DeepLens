"""Event sensor (DVS) simulation.

Simulates an event camera by computing log-intensity changes between consecutive
frames and firing events when changes exceed a contrast threshold. The output is
a dense event count map suitable for deep learning pipelines.

Reference:
    [1] Gallego et al., "Event-based Vision: A Survey", IEEE TPAMI, 2022.
    [2] Rebecq et al., "ESIM: an Open Event Camera Simulator", CoRL, 2018.
"""

import torch

from deeplens.sensor.sensor import Sensor


class EventSensor(Sensor):
    """Event sensor (Dynamic Vision Sensor) simulation.

    An event camera independently measures per-pixel log-intensity changes
    and fires an event whenever the change exceeds a contrast threshold:

        e_k = (x_k, y_k, t_k, p_k)

    where p_k ∈ {+1, -1} is the polarity. This class produces dense
    event count maps of shape (B, 2, H, W) where channel 0 counts
    positive events and channel 1 counts negative events.

    Args:
        size: Physical sensor size in mm, (width, height).
        res: Sensor resolution in pixels, (width, height).
        threshold_pos: Positive contrast threshold (C+). Default 0.2.
        threshold_neg: Negative contrast threshold (C-). Default 0.2.
        sigma_threshold: Std-dev of threshold noise. Set to 0 for
            deterministic operation. Default 0.03.
        cutoff_hz: High-pass temporal filter cutoff frequency in Hz.
            Set to 0 to disable. Default 0.
        leak_rate_hz: Leak event rate in Hz. Simulates spontaneous
            background events. Set to 0 to disable. Default 0.
        shot_noise_rate_hz: Shot noise event rate in Hz. Simulates
            noise events from dark current. Set to 0 to disable. Default 0.
        eps: Small constant added before log to avoid log(0). Default 1e-7.
    """

    def __init__(
        self,
        size=(8.0, 6.0),
        res=(4000, 3000),
        threshold_pos=0.2,
        threshold_neg=0.2,
        sigma_threshold=0.03,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        eps=1e-7,
    ):
        super().__init__(size=size, res=res)

        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.sigma_threshold = sigma_threshold
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.eps = eps

    # ------------------------------------------------------------------
    # Core event generation
    # ------------------------------------------------------------------
    def _to_gray(self, img):
        """Convert to single-channel grayscale if needed.

        Args:
            img: Tensor (B, C, H, W), range [0, 1].

        Returns:
            Tensor (B, 1, H, W).
        """
        if img.shape[1] == 3:
            # ITU-R BT.601 luma weights
            weights = img.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
            return (img * weights).sum(dim=1, keepdim=True)
        elif img.shape[1] == 1:
            return img
        else:
            raise ValueError(
                f"Expected 1 or 3 channels, got {img.shape[1]}"
            )

    def _log_intensity(self, img):
        """Compute log intensity.

        Args:
            img: Tensor (B, 1, H, W), range [0, 1].

        Returns:
            Log intensity tensor (B, 1, H, W).
        """
        return torch.log(img + self.eps)

    def forward(self, I_t, I_t_1, dt=None):
        """Convert two consecutive frames into a dense event count map.

        Each pixel independently computes the change in log intensity and
        fires integer-count positive or negative events when the change
        exceeds the contrast threshold.

        Args:
            I_t: Current frame, tensor (B, C, H, W), range [0, 1].
            I_t_1: Previous frame, tensor (B, C, H, W), range [0, 1].
            dt: Time interval between frames in seconds. Required when
                ``cutoff_hz``, ``leak_rate_hz``, or ``shot_noise_rate_hz``
                is non-zero. Default None.

        Returns:
            events: Tensor (B, 2, H, W). Channel 0 = positive event
                counts, channel 1 = negative event counts.
        """
        # Convert to grayscale
        gray_t = self._to_gray(I_t)
        gray_t_1 = self._to_gray(I_t_1)

        # Log intensity
        L_t = self._log_intensity(gray_t)
        L_t_1 = self._log_intensity(gray_t_1)

        # Temporal high-pass filter (optional)
        delta = L_t - L_t_1
        if self.cutoff_hz > 0 and dt is not None:
            tau = 1.0 / (2.0 * 3.141592653589793 * self.cutoff_hz)
            alpha = tau / (tau + dt)
            delta = delta * (1.0 - alpha)

        # Per-pixel threshold with optional noise
        theta_pos = self.threshold_pos
        theta_neg = self.threshold_neg
        if self.sigma_threshold > 0 and self.training:
            noise = torch.randn_like(delta) * self.sigma_threshold
            theta_pos = theta_pos + noise.abs()
            theta_neg = theta_neg + noise.abs()

        # Quantize into event counts
        pos_events = torch.floor(torch.clamp(delta, min=0.0) / theta_pos)
        neg_events = torch.floor(torch.clamp(-delta, min=0.0) / theta_neg)

        events = torch.cat([pos_events, neg_events], dim=1)  # (B, 2, H, W)

        # Leak noise: spontaneous background events
        if self.leak_rate_hz > 0 and dt is not None:
            leak_prob = self.leak_rate_hz * dt
            leak_mask = (torch.rand_like(pos_events) < leak_prob).float()
            events[:, 0:1] += leak_mask

        # Shot noise: random spurious events
        if self.shot_noise_rate_hz > 0 and dt is not None:
            shot_prob = self.shot_noise_rate_hz * dt
            shot_mask_pos = (torch.rand_like(pos_events) < shot_prob / 2).float()
            shot_mask_neg = (torch.rand_like(neg_events) < shot_prob / 2).float()
            events[:, 0:1] += shot_mask_pos
            events[:, 1:2] += shot_mask_neg

        return events

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------
    def forward_video(self, frames, dt=None):
        """Simulate event sensor output from a video sequence.

        Iterates over consecutive frame pairs and generates event count
        maps for each transition.

        Args:
            frames: Tensor (B, T, C, H, W), range [0, 1].
            dt: Time interval between frames in seconds. Default None.

        Returns:
            events: Tensor (B, T-1, 2, H, W). Event count maps for each
                consecutive frame pair.
        """
        B, T, C, H, W = frames.shape
        assert T >= 2, f"Need at least 2 frames, got {T}"

        event_list = []
        for t in range(1, T):
            ev = self.forward(frames[:, t], frames[:, t - 1], dt=dt)
            event_list.append(ev)

        # Stack along time dimension: (B, T-1, 2, H, W)
        return torch.stack(event_list, dim=1)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def events_to_voxel_grid(self, events, num_bins=5):
        """Convert event count map to a temporal voxel grid representation.

        Distributes events across ``num_bins`` temporal bins using linear
        interpolation. This is a common input representation for event-
        based neural networks.

        Args:
            events: Tensor (B, 2, H, W) — single-pair event counts.
            num_bins: Number of temporal bins. Default 5.

        Returns:
            voxel: Tensor (B, num_bins, H, W).
        """
        B, _, H, W = events.shape
        net_events = events[:, 0:1] - events[:, 1:2]  # (B, 1, H, W)

        voxel = net_events.expand(B, num_bins, H, W) / num_bins
        return voxel

    def events_to_timestamp_image(self, events):
        """Convert event count map to a 2-channel timestamp-like image.

        Creates a representation where non-zero pixels indicate event
        activity. Channel 0 stores positive event magnitude, channel 1
        stores negative event magnitude, both normalised to [0, 1].

        Args:
            events: Tensor (B, 2, H, W).

        Returns:
            ts_img: Tensor (B, 2, H, W), range [0, 1].
        """
        max_val = events.amax(dim=(2, 3), keepdim=True).clamp(min=1.0)
        return events / max_val
