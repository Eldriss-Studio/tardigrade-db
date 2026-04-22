"""Domain errors for benchmark framework."""


class BenchmarkError(Exception):
    """Base benchmark error."""


class ConfigError(BenchmarkError):
    """Raised when benchmark configuration is invalid."""


class DatasetUnavailableError(BenchmarkError):
    """Raised when a dataset cannot be loaded."""


class AdapterUnavailableError(BenchmarkError):
    """Raised when an external adapter service is unavailable."""
