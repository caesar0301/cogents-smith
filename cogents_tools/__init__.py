# Try to import cogents_core logging, fall back to basic logging if not available
try:
    from cogents_core.utils.logging_config import setup_logging

    # Enable colorful logging by default for cogents
    setup_logging(level="INFO", enable_colors=True)
except ImportError:
    # Fallback to basic logging if cogents_core is not available
    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Import group-wise access
from . import groups

# Import lazy loading functionality
from .lazy_import import (
    disable_lazy_loading,
    enable_lazy_loading,
    force_load_all_toolkits,
    get_available_groups,
    get_group_toolkits,
    get_loaded_modules,
    is_lazy_loading_enabled,
    load_all_toolkits,
    load_toolkit_group,
)

# Enable group-wise or toolkit-wise lazy loading by default
enable_lazy_loading()

# Make groups available at package level
__all__ = [
    "enable_lazy_loading",
    "disable_lazy_loading",
    "is_lazy_loading_enabled",
    "load_toolkit_group",
    "load_all_toolkits",
    "force_load_all_toolkits",
    "get_available_groups",
    "get_group_toolkits",
    "get_loaded_modules",
    "groups",
]
