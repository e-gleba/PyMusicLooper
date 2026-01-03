class LoopNotFoundError(Exception):
    """Raised when no valid loop points are found."""


class AudioLoadError(Exception):
    """Raised when audio file cannot be loaded or is invalid."""
