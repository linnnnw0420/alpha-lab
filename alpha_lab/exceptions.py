"""Public exception hierarchy for predictable Alpha-Lab failures."""


class AlphaLabError(Exception):
    """Base class for package-specific errors."""


class DataContractError(AlphaLabError, ValueError):
    """Input data does not satisfy the documented schema."""


class AlignmentError(AlphaLabError, ValueError):
    """Research objects cannot be aligned without ambiguity."""


class LookaheadError(AlphaLabError, ValueError):
    """A temporal ordering would leak future information."""


class ConfigurationError(AlphaLabError, ValueError):
    """A configuration value or combination is invalid."""


__all__ = [
    "AlphaLabError",
    "AlignmentError",
    "ConfigurationError",
    "DataContractError",
    "LookaheadError",
]
