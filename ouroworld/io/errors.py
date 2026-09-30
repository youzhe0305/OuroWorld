"""Exceptions raised when an on-disk artifact is missing or malformed."""


class ArtifactError(ValueError):
    """An artifact on disk does not match its documented format."""
