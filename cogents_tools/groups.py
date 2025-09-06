"""
Group-wise imports for cogents-tools.

This module provides convenient imports for semantic groups of toolkits,
allowing users to import related functionality together.
"""

from typing import Any, Dict

from .lazy_import import get_available_groups, get_group_toolkits, lazy_import, load_toolkit_group


class ToolkitGroup:
    """A container for a group of related toolkits."""

    def __init__(self, group_name: str):
        self.group_name = group_name
        self._toolkits = None

    @property
    def toolkits(self) -> Dict[str, Any]:
        """Get the toolkits in this group (lazy loaded)."""
        if self._toolkits is None:
            self._toolkits = load_toolkit_group(self.group_name)
        return self._toolkits

    def __getattr__(self, name: str):
        """Get a toolkit by name."""
        if name in self.toolkits:
            return self.toolkits[name]
        raise AttributeError(f"Toolkit '{name}' not found in group '{self.group_name}'")

    def __dir__(self):
        """Return available toolkit names."""
        return list(self.toolkits.keys())

    def __repr__(self):
        toolkit_names = get_group_toolkits(self.group_name)
        return f"ToolkitGroup('{self.group_name}', toolkits={toolkit_names})"


# Create group instances
@lazy_import
def academic() -> ToolkitGroup:
    """
    Academic research toolkits.

    Includes:
    - arxiv_toolkit: Search and download academic papers from arXiv

    Example:
        >>> from cogents_tools.groups import academic
        >>> arxiv = academic().arxiv_toolkit()
        >>> papers = await arxiv.search_papers("machine learning", max_results=5)
    """
    return ToolkitGroup("academic")


@lazy_import
def image() -> ToolkitGroup:
    """
    Image processing and analysis toolkits.

    Includes:
    - image_toolkit: Image analysis, visual question answering, and processing

    Example:
        >>> from cogents_tools.groups import image
        >>> img_toolkit = image().image_toolkit()
        >>> result = await img_toolkit.analyze_image("path/to/image.jpg", "What's in this image?")
    """
    return ToolkitGroup("image")


@lazy_import
def video() -> ToolkitGroup:
    """
    Video processing and analysis toolkits.

    Includes:
    - video_toolkit: Video processing and analysis capabilities

    Example:
        >>> from cogents_tools.groups import video
        >>> vid_toolkit = video().video_toolkit()
    """
    return ToolkitGroup("video")


@lazy_import
def audio() -> ToolkitGroup:
    """
    Audio processing and analysis toolkits.

    Includes:
    - audio_toolkit: General audio processing capabilities
    - audio_aliyun_toolkit: Aliyun-specific audio processing

    Example:
        >>> from cogents_tools.groups import audio
        >>> audio_tools = audio()
        >>> general_audio = audio_tools.audio_toolkit()
        >>> aliyun_audio = audio_tools.audio_aliyun_toolkit()
    """
    return ToolkitGroup("audio")


@lazy_import
def development() -> ToolkitGroup:
    """
    Development and programming toolkits.

    Includes:
    - bash_toolkit: Execute bash commands safely
    - file_edit_toolkit: File editing and manipulation
    - github_toolkit: GitHub API integration
    - python_executor_toolkit: Python code execution
    - tabular_data_toolkit: Work with tabular data (CSV, Excel, etc.)

    Example:
        >>> from cogents_tools.groups import development
        >>> dev_tools = development()
        >>> bash = dev_tools.bash_toolkit()
        >>> result = await bash.execute_command("ls -la")
    """
    return ToolkitGroup("development")


@lazy_import
def file_processing() -> ToolkitGroup:
    """
    File processing and manipulation toolkits.

    Includes:
    - document_toolkit: Process various document formats
    - file_edit_toolkit: File editing and manipulation
    - tabular_data_toolkit: Work with tabular data (CSV, Excel, etc.)

    Example:
        >>> from cogents_tools.groups import file_processing
        >>> file_tools = file_processing()
        >>> doc_toolkit = file_tools.document_toolkit()
        >>> content = await doc_toolkit.extract_text("document.pdf")
    """
    return ToolkitGroup("file_processing")


@lazy_import
def communication() -> ToolkitGroup:
    """
    Communication and messaging toolkits.

    Includes:
    - memory_toolkit: Memory management and persistence

    Example:
        >>> from cogents_tools.groups import communication
        >>> comm_tools = communication()
        >>> memory = comm_tools.memory_toolkit()
    """
    return ToolkitGroup("communication")


@lazy_import
def info_retrieval() -> ToolkitGroup:
    """
    Information retrieval and search toolkits.

    Includes:
    - search_toolkit: General web search capabilities
    - serper_toolkit: Serper API for search
    - wikipedia_toolkit: Wikipedia search and content retrieval

    Example:
        >>> from cogents_tools.groups import info_retrieval
        >>> search_tools = info_retrieval()
        >>> wiki = search_tools.wikipedia_toolkit()
        >>> content = await wiki.search("artificial intelligence")
    """
    return ToolkitGroup("info_retrieval")


@lazy_import
def persistence() -> ToolkitGroup:
    """
    Data persistence and storage toolkits.

    Includes:
    - memory_toolkit: Memory management and persistence

    Example:
        >>> from cogents_tools.groups import persistence
        >>> persist_tools = persistence()
        >>> memory = persist_tools.memory_toolkit()
    """
    return ToolkitGroup("persistence")


@lazy_import
def hitl() -> ToolkitGroup:
    """
    Human-in-the-loop (HITL) interaction toolkits.

    Includes:
    - user_interaction_toolkit: Interactive user communication

    Example:
        >>> from cogents_tools.groups import hitl
        >>> hitl_tools = hitl()
        >>> user_interaction = hitl_tools.user_interaction_toolkit()
        >>> response = await user_interaction.ask_user("What would you like to do?")
    """
    return ToolkitGroup("hitl")


# Convenience function to list all available groups
def list_groups() -> Dict[str, str]:
    """
    List all available toolkit groups with descriptions.

    Returns:
        Dict mapping group names to their descriptions.
    """
    return {
        "academic": "Academic research toolkits (arxiv)",
        "image": "Image processing and analysis toolkits",
        "video": "Video processing and analysis toolkits",
        "audio": "Audio processing toolkits (general + Aliyun)",
        "development": "Development tools (bash, file editing, GitHub, Python execution, tabular data)",
        "file_processing": "File processing toolkits (documents, file editing, tabular data)",
        "communication": "Communication and messaging toolkits",
        "info_retrieval": "Information retrieval and search toolkits (web search, Wikipedia)",
        "persistence": "Data persistence and storage toolkits",
        "hitl": "Human-in-the-loop interaction toolkits",
    }


def get_group_info(group_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a toolkit group.

    Args:
        group_name: Name of the group to get info for

    Returns:
        Dictionary with group information including toolkits list
    """
    if group_name not in get_available_groups():
        raise ValueError(f"Unknown group: {group_name}")

    toolkits = get_group_toolkits(group_name)
    descriptions = list_groups()

    return {
        "name": group_name,
        "description": descriptions.get(group_name, "No description available"),
        "toolkits": toolkits,
        "toolkit_count": len(toolkits),
    }
