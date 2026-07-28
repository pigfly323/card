"""
Error codes and constants for the environment platform.
"""

# =============================================================================
# Parameter Validation Error Codes
# =============================================================================

MISSING_PARAM = "MISSING_PARAM"
INVALID_PARAMS = "INVALID_PARAMS"
PARAM_TYPE_MISMATCH = "PARAM_TYPE_MISMATCH"
PARAM_SCHEMA_VIOLATION = "PARAM_SCHEMA_VIOLATION"
PYTHON_ERROR = "PYTHON_ERROR"

# =============================================================================
# Random Failure Error Codes
# =============================================================================

SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
NETWORK_ERROR = "NETWORK_ERROR"
RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

# All random failure error types
RANDOM_FAILURE_ERROR_TYPES = [
    SERVICE_UNAVAILABLE,
    NETWORK_TIMEOUT,
    NETWORK_ERROR,
    RATE_LIMIT_EXCEEDED,
    INTERNAL_SERVER_ERROR,
]

# =============================================================================
# Memory Safety Error Codes
# =============================================================================

OOM_SIGNAL_ERROR = "OOM_SIGNAL_ERROR"
MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"

# =============================================================================
# Exceptions
# =============================================================================


class EnvPlatformError(Exception):
    """Base exception for environment platform errors."""
    pass


class SessionNotFoundError(EnvPlatformError):
    """Raised when a session is not found."""
    pass


class TaskNotFoundError(EnvPlatformError):
    """Raised when a task is not found."""
    pass


class ToolNotFoundError(EnvPlatformError):
    """Raised when a tool is not found."""
    pass
