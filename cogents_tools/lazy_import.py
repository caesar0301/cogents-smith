"""
Lazy importing mechanism for cogents-tools.

This module provides utilities for lazy loading of toolkits and their dependencies,
allowing for faster import times and reduced memory usage by only loading modules
when they are actually needed.
"""

import importlib
import logging
import warnings
from functools import wraps
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Global registry for lazy imports
_lazy_registry: Dict[str, Dict[str, Any]] = {}
_loaded_modules: Set[str] = set()
_lazy_enabled = False

# Toolkit group definitions
TOOLKIT_GROUPS = {
    "academic": ["arxiv_toolkit"],
    "image": ["image_toolkit"],
    "video": ["video_toolkit"],
    "audio": [
        "audio_toolkit",
        "audio_aliyun_toolkit",
    ],
    "development": [
        "bash_toolkit",
        "file_edit_toolkit",
        "github_toolkit",
        "python_executor_toolkit",
        "tabular_data_toolkit",
    ],
    "file_processing": [
        "document_toolkit",
        "file_edit_toolkit",
        "tabular_data_toolkit",
    ],
    "communication": ["gmail_toolkit"],
    "info_retrieval": [
        "search_toolkit",
        "serper_toolkit",
        "wikipedia_toolkit",
    ],
    "memorization": ["memory_toolkit"],
    "hitl": ["user_interaction_toolkit"],  # human in the loop
}

# Reverse mapping for toolkit to groups
TOOLKIT_TO_GROUPS = {}
for group, toolkits in TOOLKIT_GROUPS.items():
    for toolkit in toolkits:
        if toolkit not in TOOLKIT_TO_GROUPS:
            TOOLKIT_TO_GROUPS[toolkit] = []
        TOOLKIT_TO_GROUPS[toolkit].append(group)


class LazyModule:
    """A proxy object that loads a module only when accessed."""

    def __init__(self, module_name: str, package: Optional[str] = None):
        self._module_name = module_name
        self._package = package
        self._module = None
        self._loaded = False

    def _load_module(self):
        """Load the actual module."""
        if not self._loaded:
            try:
                logger.debug(f"Lazy loading module: {self._module_name}")
                self._module = importlib.import_module(self._module_name, self._package)
                self._loaded = True
                _loaded_modules.add(self._module_name)
            except ImportError as e:
                logger.warning(f"Failed to lazy load {self._module_name}: {e}")
                raise
        return self._module

    def __getattr__(self, name: str):
        """Get attribute from the loaded module."""
        module = self._load_module()
        return getattr(module, name)

    def __dir__(self):
        """Return directory of the loaded module."""
        try:
            module = self._load_module()
            return dir(module)
        except ImportError:
            return []


class LazyToolkit:
    """A proxy object that loads a toolkit only when accessed."""

    def __init__(self, toolkit_name: str, module_path: str):
        self._toolkit_name = toolkit_name
        self._module_path = module_path
        self._toolkit_class = None
        self._loaded = False

    def _is_toolkit_class(self, cls):
        """
        Check if a class is a valid toolkit class.

        Simple criteria:
        1. Must be a valid class
        2. Must inherit from AsyncBaseToolkit
        3. Must end with "Toolkit" (naming convention)
        4. Must be defined in the current module (not imported)
        """
        return (
            self._is_valid_class(cls)
            and self._inherits_from_base_toolkit(cls)
            and cls.__name__.endswith("Toolkit")
            and not cls.__name__.startswith("Base")  # Exclude base classes
            and hasattr(cls, "__module__")
            and cls.__module__ == self._module_path
        )

    def _is_valid_class(self, obj) -> bool:
        """Check if object is a valid class."""
        import inspect

        return (
            inspect.isclass(obj)
            and hasattr(obj, "__name__")
            and hasattr(obj, "__bases__")
            and obj.__name__ != "type"  # Exclude metaclasses
        )

    def _inherits_from_base_toolkit(self, cls) -> bool:
        """Check if class inherits from AsyncBaseToolkit or similar base class."""
        import inspect

        # Get the method resolution order (MRO) to check all parent classes
        mro = inspect.getmro(cls)

        # Look for common base toolkit class names
        base_toolkit_names = {"AsyncBaseToolkit", "BaseToolkit"}

        for base_class in mro:
            if base_class.__name__ in base_toolkit_names:
                return True
            # Also check module path for cogents_core toolify base classes
            if hasattr(base_class, "__module__") and base_class.__module__:
                if "cogents_core.toolify.base" in base_class.__module__:
                    if base_class.__name__.endswith(("Toolkit", "Base")):
                        return True

        return False

    def _get_expected_class_name(self) -> str:
        """Get the expected class name based on toolkit name."""
        # Convert snake_case toolkit name to CamelCase class name
        # e.g., "audio_aliyun_toolkit" -> "AudioAliyunToolkit"
        parts = self._toolkit_name.replace("_toolkit", "").split("_")
        class_name = "".join(word.capitalize() for word in parts) + "Toolkit"
        return class_name

    def _load_toolkit(self):
        """Load the actual toolkit class."""
        if not self._loaded:
            try:
                logger.debug(f"Lazy loading toolkit: {self._toolkit_name}")
                module = importlib.import_module(self._module_path)

                # Find toolkit class using simple, reliable approach
                candidates = []

                for attr_name in dir(module):
                    try:
                        attr = getattr(module, attr_name)
                        if self._is_toolkit_class(attr):
                            candidates.append(attr)
                            logger.debug(f"Found toolkit candidate: {attr.__name__}")
                    except Exception:
                        # Skip attributes that can't be accessed
                        continue

                # Select the toolkit class
                if len(candidates) == 1:
                    self._toolkit_class = candidates[0]
                elif len(candidates) > 1:
                    # If multiple candidates, prefer exact name match
                    expected_class_name = self._get_expected_class_name()
                    for candidate in candidates:
                        if candidate.__name__ == expected_class_name:
                            self._toolkit_class = candidate
                            break
                    else:
                        # Fallback to first candidate
                        self._toolkit_class = candidates[0]
                        logger.debug(f"Multiple candidates found, using: {self._toolkit_class.__name__}")
                else:
                    # No candidates found
                    raise ImportError(
                        f"No toolkit class found in {self._module_path} for {self._toolkit_name}. "
                        f"Expected a class inheriting from BaseToolkit with @register_toolkit decorator."
                    )

                logger.debug(f"Selected toolkit class: {self._toolkit_class.__name__}")
                self._loaded = True
                _loaded_modules.add(self._module_path)

            except ImportError as e:
                logger.warning(f"Failed to lazy load toolkit {self._toolkit_name}: {e}")
                raise
        return self._toolkit_class

    def __call__(self, *args, **kwargs):
        """Create an instance of the toolkit."""
        toolkit_class = self._load_toolkit()
        return toolkit_class(*args, **kwargs)

    def __getattr__(self, name: str):
        """Get attribute from the loaded toolkit class."""
        toolkit_class = self._load_toolkit()
        return getattr(toolkit_class, name)


def enable_lazy_loading():
    """Enable lazy loading globally."""
    global _lazy_enabled
    _lazy_enabled = True
    logger.info("Lazy loading enabled for cogents-tools")


def disable_lazy_loading():
    """Disable lazy loading globally."""
    global _lazy_enabled
    _lazy_enabled = False
    logger.info("Lazy loading disabled for cogents-tools")


def is_lazy_loading_enabled() -> bool:
    """Check if lazy loading is enabled."""
    return _lazy_enabled


def register_lazy_module(name: str, module_path: str, package: Optional[str] = None):
    """Register a module for lazy loading."""
    if name not in _lazy_registry:
        _lazy_registry[name] = {}

    _lazy_registry[name]["module"] = LazyModule(module_path, package)
    logger.debug(f"Registered lazy module: {name} -> {module_path}")


def register_lazy_toolkit(name: str, module_path: str):
    """Register a toolkit for lazy loading."""
    if name not in _lazy_registry:
        _lazy_registry[name] = {}

    _lazy_registry[name]["toolkit"] = LazyToolkit(name, module_path)
    logger.debug(f"Registered lazy toolkit: {name} -> {module_path}")


def get_lazy_module(name: str) -> Optional[LazyModule]:
    """Get a lazy module by name."""
    if name in _lazy_registry and "module" in _lazy_registry[name]:
        return _lazy_registry[name]["module"]
    return None


def get_lazy_toolkit(name: str) -> Optional[LazyToolkit]:
    """Get a lazy toolkit by name."""
    if name in _lazy_registry and "toolkit" in _lazy_registry[name]:
        return _lazy_registry[name]["toolkit"]
    return None


def load_toolkit_group(group_name: str) -> Dict[str, Any]:
    """Load all toolkits in a semantic group."""
    if group_name not in TOOLKIT_GROUPS:
        raise ValueError(f"Unknown toolkit group: {group_name}. Available groups: {list(TOOLKIT_GROUPS.keys())}")

    loaded_toolkits = {}
    toolkits = TOOLKIT_GROUPS[group_name]

    logger.info(f"Loading toolkit group '{group_name}' with toolkits: {toolkits}")

    for toolkit_name in toolkits:
        try:
            lazy_toolkit = get_lazy_toolkit(toolkit_name)
            if lazy_toolkit:
                loaded_toolkits[toolkit_name] = lazy_toolkit
            else:
                logger.warning(f"Toolkit {toolkit_name} not found in lazy registry")
        except Exception as e:
            logger.error(f"Failed to load toolkit {toolkit_name}: {e}")

    return loaded_toolkits


def load_all_toolkits() -> Dict[str, Any]:
    """Load all registered toolkits."""
    loaded_toolkits = {}

    for name, registry_entry in _lazy_registry.items():
        if "toolkit" in registry_entry:
            try:
                loaded_toolkits[name] = registry_entry["toolkit"]
            except Exception as e:
                logger.error(f"Failed to load toolkit {name}: {e}")

    return loaded_toolkits


def force_load_all_toolkits() -> Dict[str, Any]:
    """Force load all registered toolkits by actually accessing their classes."""
    loaded_toolkits = {}

    for name, registry_entry in _lazy_registry.items():
        if "toolkit" in registry_entry:
            try:
                lazy_toolkit = registry_entry["toolkit"]
                # Force loading by accessing the toolkit class
                toolkit_class = lazy_toolkit._load_toolkit()
                loaded_toolkits[name] = toolkit_class
                logger.debug(f"Force loaded toolkit: {name} -> {toolkit_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to force load toolkit {name}: {e}")

    return loaded_toolkits


def get_loaded_modules() -> Set[str]:
    """Get the set of modules that have been loaded."""
    return _loaded_modules.copy()


def get_available_groups() -> List[str]:
    """Get list of available toolkit groups."""
    return list(TOOLKIT_GROUPS.keys())


def get_group_toolkits(group_name: str) -> List[str]:
    """Get list of toolkits in a group."""
    return TOOLKIT_GROUPS.get(group_name, [])


def lazy_import(func):
    """Decorator to enable lazy importing for a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _lazy_enabled:
            warnings.warn("Lazy loading is not enabled. Call enable_lazy_loading() first.", UserWarning)
        return func(*args, **kwargs)

    return wrapper


# Initialize lazy loading for all toolkits
def _initialize_lazy_toolkits():
    """Initialize lazy loading for all available toolkits."""
    toolkit_modules = {
        "arxiv_toolkit": "cogents_tools.toolkits.arxiv_toolkit",
        "audio_aliyun_toolkit": "cogents_tools.toolkits.audio_aliyun_toolkit",
        "audio_toolkit": "cogents_tools.toolkits.audio_toolkit",
        "bash_toolkit": "cogents_tools.toolkits.bash_toolkit",
        "document_toolkit": "cogents_tools.toolkits.document_toolkit",
        "file_edit_toolkit": "cogents_tools.toolkits.file_edit_toolkit",
        "github_toolkit": "cogents_tools.toolkits.github_toolkit",
        "gmail_toolkit": "cogents_tools.toolkits.gmail_toolkit",
        "image_toolkit": "cogents_tools.toolkits.image_toolkit",
        "memory_toolkit": "cogents_tools.toolkits.memory_toolkit",
        "python_executor_toolkit": "cogents_tools.toolkits.python_executor_toolkit",
        "search_toolkit": "cogents_tools.toolkits.search_toolkit",
        "serper_toolkit": "cogents_tools.toolkits.serper_toolkit",
        "tabular_data_toolkit": "cogents_tools.toolkits.tabular_data_toolkit",
        "user_interaction_toolkit": "cogents_tools.toolkits.user_interaction_toolkit",
        "video_toolkit": "cogents_tools.toolkits.video_toolkit",
        "wikipedia_toolkit": "cogents_tools.toolkits.wikipedia_toolkit",
    }

    for toolkit_name, module_path in toolkit_modules.items():
        register_lazy_toolkit(toolkit_name, module_path)


# Initialize on import
_initialize_lazy_toolkits()
