"""Event sensor stub."""

from deeplens.sensor.sensor import Sensor


class EventSensor(Sensor):
    """Event sensor."""

    def __init__(self, size=(8.0, 6.0), res=(4000, 3000)):
        super().__init__(size=size, res=res)

    def forward(self, I_t, I_t_1):
        """Converts light illuminance to event stream.

        Args:
            I_t: Current frame
            I_t_1: Previous frame

        Returns:
            Event stream
        """
        pass

    def forward_video(self, frames):
        """Simulate sensor output from a video.

        Args:
            frames: Tensor of shape (B, T, 3, H, W), range [0, 1]

        Returns:
            Event stream for the video sequence
        """
        pass
