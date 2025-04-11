import torch

from .sensor import Sensor

class EventSensor(Sensor):
    """Event sensor"""

    def __init__(self, bit=10, black_level=64):
        super().__init__(bit, black_level)

    def forward(self, I_t, I_t_1):
        """Converts light illuminance to event stream.

        Args:
            I_t: Current frame
            I_t_1: Previous frame

        Returns:
            Event stream
        """
        # Converts light illuminance to event stream.
        pass

    def forward_video(self, frames):
        """Simulate sensor output from a video.

        Args:
            frames: Tensor of shape (B, T, 3, H, W), range [0, 1]

        Returns:
            Event stream for the video sequence
        """
        pass
