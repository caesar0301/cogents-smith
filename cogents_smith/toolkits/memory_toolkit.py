"""
Memory toolkit for persistent text storage and manipulation.

Provides tools for storing, retrieving, and editing persistent text content
with safety features and comprehensive error handling.
"""

import os
from pathlib import Path
from typing import Callable, Dict, Optional

from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


@register_toolkit("memory")
class MemoryToolkit(AsyncBaseToolkit):
    """
    Toolkit for persistent memory storage and manipulation.

    This toolkit provides capabilities for:
    - Storing and retrieving persistent text content
    - Editing memory content with string replacement
    - Multiple memory slots for different contexts
    - File-based persistence across sessions
    - Safety warnings for overwrite operations
    - Search and pattern matching in memory

    Features:
    - In-memory and file-based storage options
    - Multiple named memory slots
    - String replacement with occurrence counting
    - Content validation and safety checks
    - Backup and versioning support
    - Search and filtering capabilities

    Use cases:
    - Maintaining conversation context
    - Storing intermediate results
    - Building knowledge bases
    - Caching computed information
    - Maintaining state across operations
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the memory toolkit.

        Args:
            config: Toolkit configuration
        """
        super().__init__(config)

        # Configuration
        self.storage_type = self.config.config.get("storage_type", "memory")  # "memory" or "file"
        self.storage_dir = Path(self.config.config.get("storage_dir", "./memory_storage"))
        self.max_memory_size = self.config.config.get("max_memory_size", 1024 * 1024)  # 1MB
        self.enable_versioning = self.config.config.get("enable_versioning", False)

        # In-memory storage
        self.memory_slots: Dict[str, str] = {}

        # File storage setup
        if self.storage_type == "file":
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_memory()

        self.logger.info(f"Memory toolkit initialized with {self.storage_type} storage")

    def _get_memory_file_path(self, slot_name: str) -> Path:
        """Get file path for a memory slot."""
        safe_name = "".join(c for c in slot_name if c.isalnum() or c in "._-")
        return self.storage_dir / f"{safe_name}.txt"

    def _load_persistent_memory(self):
        """Load memory from persistent storage."""
        try:
            for file_path in self.storage_dir.glob("*.txt"):
                slot_name = file_path.stem
                with open(file_path, "r", encoding="utf-8") as f:
                    self.memory_slots[slot_name] = f.read()

            self.logger.info(f"Loaded {len(self.memory_slots)} memory slots from storage")

        except Exception as e:
            self.logger.warning(f"Failed to load persistent memory: {e}")

    def _save_memory_slot(self, slot_name: str, content: str):
        """Save a memory slot to persistent storage."""
        if self.storage_type != "file":
            return

        try:
            file_path = self._get_memory_file_path(slot_name)

            # Create backup if versioning is enabled
            if self.enable_versioning and file_path.exists():
                backup_path = file_path.with_suffix(f".{int(os.path.getmtime(file_path))}.bak")
                file_path.rename(backup_path)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            self.logger.error(f"Failed to save memory slot '{slot_name}': {e}")

    def _validate_content_size(self, content: str) -> bool:
        """Validate that content doesn't exceed size limits."""
        return len(content.encode("utf-8")) <= self.max_memory_size

    async def read_memory(self, slot_name: str = "default") -> str:
        """
        Read the contents of a memory slot.

        This tool retrieves the current content stored in the specified memory slot.
        Memory slots allow you to maintain separate contexts or information stores.

        Args:
            slot_name: Name of the memory slot to read (default: "default")

        Returns:
            Current content of the memory slot, or empty string if slot doesn't exist

        Example:
            content = await read_memory("conversation_context")
            print(f"Current context: {content}")
        """
        self.logger.info(f"Reading memory slot: {slot_name}")

        content = self.memory_slots.get(slot_name, "")

        if not content:
            return f"Memory slot '{slot_name}' is empty or does not exist."

        self.logger.info(f"Read {len(content)} characters from slot '{slot_name}'")
        return content

    async def write_memory(self, content: str, slot_name: str = "default") -> str:
        """
        Write content to a memory slot, replacing any existing content.

        This tool stores content in the specified memory slot. If the slot already
        contains content, it will be completely replaced. A warning is provided
        when overwriting existing content.

        Args:
            content: Content to store in memory
            slot_name: Name of the memory slot to write to (default: "default")

        Returns:
            Success message, including warning if overwriting existing content

        Example:
            result = await write_memory("Important information to remember", "notes")
        """
        self.logger.info(f"Writing to memory slot: {slot_name}")

        # Validate content size
        if not self._validate_content_size(content):
            return f"Error: Content too large ({len(content)} chars, max: {self.max_memory_size})"

        # Check if overwriting existing content
        existing_content = self.memory_slots.get(slot_name, "")
        warning_msg = ""

        if existing_content:
            warning_msg = (
                f"Warning: Overwriting existing content in slot '{slot_name}'. "
                f"Previous content ({len(existing_content)} chars) was:\n"
                f"{existing_content[:200]}{'...' if len(existing_content) > 200 else ''}\n\n"
            )

        # Store the content
        self.memory_slots[slot_name] = content

        # Save to persistent storage if enabled
        self._save_memory_slot(slot_name, content)

        result_msg = f"Memory slot '{slot_name}' updated successfully with {len(content)} characters."

        self.logger.info(f"Wrote {len(content)} characters to slot '{slot_name}'")
        return warning_msg + result_msg

    async def edit_memory(self, old_string: str, new_string: str, slot_name: str = "default") -> str:
        """
        Edit memory content by replacing occurrences of a string.

        This tool performs string replacement within a memory slot. It provides
        safety checks for multiple occurrences and clear feedback about changes made.

        Args:
            old_string: String to find and replace
            new_string: String to replace with
            slot_name: Name of the memory slot to edit (default: "default")

        Returns:
            Result message indicating success, failure, or warnings

        Example:
            result = await edit_memory("old info", "new info", "notes")
        """
        self.logger.info(f"Editing memory slot: {slot_name}")

        # Check if slot exists
        if slot_name not in self.memory_slots:
            return f"Error: Memory slot '{slot_name}' does not exist."

        current_content = self.memory_slots[slot_name]

        # Check if old_string exists
        if old_string not in current_content:
            return f"Error: String '{old_string}' not found in memory slot '{slot_name}'."

        # Count occurrences
        occurrence_count = current_content.count(old_string)

        if occurrence_count > 1:
            return (
                f"Warning: Found {occurrence_count} occurrences of '{old_string}' "
                f"in slot '{slot_name}'. Please use more specific context to avoid "
                "unintended replacements, or use replace_all_in_memory for intentional "
                "multiple replacements."
            )

        # Perform replacement
        new_content = current_content.replace(old_string, new_string, 1)

        # Validate new content size
        if not self._validate_content_size(new_content):
            return f"Error: Edited content would be too large"

        # Update memory
        self.memory_slots[slot_name] = new_content
        self._save_memory_slot(slot_name, new_content)

        self.logger.info(f"Edited memory slot '{slot_name}': replaced 1 occurrence")
        return f"Successfully replaced 1 occurrence of '{old_string}' with '{new_string}' in slot '{slot_name}'."

    async def append_to_memory(self, content: str, slot_name: str = "default", separator: str = "\n") -> str:
        """
        Append content to an existing memory slot.

        Args:
            content: Content to append
            slot_name: Name of the memory slot (default: "default")
            separator: Separator to use between existing and new content

        Returns:
            Success message
        """
        existing_content = self.memory_slots.get(slot_name, "")

        if existing_content:
            new_content = existing_content + separator + content
        else:
            new_content = content

        if not self._validate_content_size(new_content):
            return f"Error: Combined content would be too large"

        self.memory_slots[slot_name] = new_content
        self._save_memory_slot(slot_name, new_content)

        self.logger.info(f"Appended {len(content)} characters to slot '{slot_name}'")
        return f"Successfully appended content to memory slot '{slot_name}'."

    async def clear_memory(self, slot_name: str = "default") -> str:
        """
        Clear the contents of a memory slot.

        Args:
            slot_name: Name of the memory slot to clear (default: "default")

        Returns:
            Success message
        """
        if slot_name in self.memory_slots:
            del self.memory_slots[slot_name]

            # Remove from persistent storage
            if self.storage_type == "file":
                file_path = self._get_memory_file_path(slot_name)
                if file_path.exists():
                    file_path.unlink()

            self.logger.info(f"Cleared memory slot: {slot_name}")
            return f"Memory slot '{slot_name}' has been cleared."
        else:
            return f"Memory slot '{slot_name}' does not exist."

    async def list_memory_slots(self) -> str:
        """
        List all available memory slots with their sizes.

        Returns:
            Formatted list of memory slots
        """
        if not self.memory_slots:
            return "No memory slots exist."

        slot_info = []
        for slot_name, content in self.memory_slots.items():
            size = len(content)
            preview = content[:50].replace("\n", " ") if content else "(empty)"
            if len(content) > 50:
                preview += "..."

            slot_info.append(f"  {slot_name}: {size} chars - {preview}")

        result = f"Memory slots ({len(self.memory_slots)} total):\n" + "\n".join(slot_info)
        return result

    async def search_memory(self, query: str, slot_name: Optional[str] = None) -> str:
        """
        Search for text within memory slots.

        Args:
            query: Text to search for
            slot_name: Specific slot to search (if None, searches all slots)

        Returns:
            Search results with context
        """
        results = []

        slots_to_search = {slot_name: self.memory_slots[slot_name]} if slot_name else self.memory_slots

        for name, content in slots_to_search.items():
            if query.lower() in content.lower():
                # Find all occurrences with context
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if query.lower() in line.lower():
                        context_start = max(0, i - 1)
                        context_end = min(len(lines), i + 2)
                        context_lines = lines[context_start:context_end]

                        results.append(f"Slot '{name}', line {i+1}:")
                        results.extend(f"  {j+context_start+1}: {line}" for j, line in enumerate(context_lines))
                        results.append("")

        if not results:
            search_scope = f"slot '{slot_name}'" if slot_name else "all memory slots"
            return f"No matches found for '{query}' in {search_scope}."

        return "\n".join(results)

    async def get_memory_stats(self) -> str:
        """
        Get statistics about memory usage.

        Returns:
            Formatted memory statistics
        """
        total_slots = len(self.memory_slots)
        total_chars = sum(len(content) for content in self.memory_slots.values())
        total_bytes = sum(len(content.encode("utf-8")) for content in self.memory_slots.values())

        if total_slots == 0:
            return "No memory slots exist."

        avg_size = total_chars // total_slots
        largest_slot = max(self.memory_slots.items(), key=lambda x: len(x[1]))

        stats = [
            f"Memory Statistics:",
            f"  Total slots: {total_slots}",
            f"  Total characters: {total_chars:,}",
            f"  Total bytes: {total_bytes:,}",
            f"  Average slot size: {avg_size:,} characters",
            f"  Largest slot: '{largest_slot[0]}' ({len(largest_slot[1]):,} chars)",
            f"  Storage type: {self.storage_type}",
            f"  Max slot size: {self.max_memory_size:,} bytes",
        ]

        return "\n".join(stats)

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "read_memory": self.read_memory,
            "write_memory": self.write_memory,
            "edit_memory": self.edit_memory,
            "append_to_memory": self.append_to_memory,
            "clear_memory": self.clear_memory,
            "list_memory_slots": self.list_memory_slots,
            "search_memory": self.search_memory,
            "get_memory_stats": self.get_memory_stats,
        }
