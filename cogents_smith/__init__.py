# Try to import cogents_core logging, fall back to basic logging if not available
try:
    from cogents_core.utils.logging import setup_logging

    # Enable colorful logging by default for cogents
    setup_logging(level="INFO", enable_colors=True)
except ImportError:
    # Fallback to basic logging if cogents_core is not available
    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Import group-wise access
from . import groups

# Import group functionality
from .groups import (
    get_available_groups,
    get_group_toolkits,
    load_toolkit_group,
)

# Make groups available at package level
__all__ = [
    "load_toolkit_group",
    "get_available_groups",
    "get_group_toolkits",
    "groups",
]
