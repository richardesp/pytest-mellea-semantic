"""Package-specific exceptions."""


class SemanticAssertionRuntimeError(RuntimeError):
    """Raised when a semantic assertion cannot be evaluated.

    Args:
        message: Human-readable failure message with remediation steps.
    """
