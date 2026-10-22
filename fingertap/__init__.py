"""Webcam-based finger-tapping measurement package.

Runtime pipeline (spec sheet section 1.2):
    video -> MediaPipe landmarks -> normalized distance -> causal smoothing
          -> tap state machine -> metrics/export

This is a computer-vision measurement and evaluation prototype for coursework
and research. It is NOT a diagnostic system or medical device.
"""

__version__ = "0.1.0"
