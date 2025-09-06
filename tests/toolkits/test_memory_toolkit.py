"""
Tests for MemoryToolkit functionality.
"""

import tempfile
from pathlib import Path

import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def memory_config():
    """Create a test configuration for MemoryToolkit."""
    return ToolkitConfig(name="memory", config={"storage_type": "memory", "max_memory_size": 1000})


@pytest.fixture
def file_memory_config():
    """Create a test configuration for file-based MemoryToolkit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield ToolkitConfig(
            name="memory", config={"storage_type": "file", "storage_dir": temp_dir, "max_memory_size": 1000}
        )


@pytest.fixture
def memory_toolkit(memory_config):
    """Create MemoryToolkit instance for testing."""
    return get_toolkit("memory", memory_config)


@pytest.fixture
def file_memory_toolkit(file_memory_config):
    """Create file-based MemoryToolkit instance for testing."""
    return get_toolkit("memory", file_memory_config)


class TestMemoryToolkit:
    """Test cases for MemoryToolkit."""

    async def test_toolkit_initialization(self, memory_toolkit):
        """Test that MemoryToolkit initializes correctly."""
        assert memory_toolkit is not None
        assert hasattr(memory_toolkit, "read_memory")
        assert hasattr(memory_toolkit, "write_memory")
        assert hasattr(memory_toolkit, "edit_memory")

    async def test_get_tools_map(self, memory_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await memory_toolkit.get_tools_map()

        expected_tools = [
            "read_memory",
            "write_memory",
            "edit_memory",
            "append_to_memory",
            "clear_memory",
            "list_memory_slots",
            "search_memory",
            "get_memory_stats",
        ]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    async def test_write_and_read_memory(self, memory_toolkit):
        """Test basic write and read operations."""
        # Write to memory
        content = "This is test content for memory."
        result = await memory_toolkit.write_memory(content, "test_slot")

        assert isinstance(result, str)
        assert "updated successfully" in result

        # Read from memory
        read_result = await memory_toolkit.read_memory("test_slot")

        assert read_result == content

    async def test_read_empty_slot(self, memory_toolkit):
        """Test reading from empty memory slot."""
        result = await memory_toolkit.read_memory("nonexistent_slot")

        assert isinstance(result, str)
        assert "empty or does not exist" in result

    async def test_overwrite_warning(self, memory_toolkit):
        """Test overwrite warning when replacing content."""
        # Write initial content
        await memory_toolkit.write_memory("Initial content", "test_slot")

        # Overwrite with new content
        result = await memory_toolkit.write_memory("New content", "test_slot")

        assert isinstance(result, str)
        assert "Warning: Overwriting existing content" in result
        assert "Initial content" in result

        # Verify new content
        read_result = await memory_toolkit.read_memory("test_slot")
        assert read_result == "New content"

    async def test_edit_memory_success(self, memory_toolkit):
        """Test successful memory editing."""
        # Write initial content
        await memory_toolkit.write_memory("Hello world, this is a test.", "test_slot")

        # Edit content
        result = await memory_toolkit.edit_memory("world", "universe", "test_slot")

        assert isinstance(result, str)
        assert "Successfully replaced 1 occurrence" in result

        # Verify edited content
        read_result = await memory_toolkit.read_memory("test_slot")
        assert read_result == "Hello universe, this is a test."

    async def test_edit_memory_not_found(self, memory_toolkit):
        """Test editing with string not found."""
        await memory_toolkit.write_memory("Hello world", "test_slot")

        result = await memory_toolkit.edit_memory("universe", "cosmos", "test_slot")

        assert isinstance(result, str)
        assert "not found in memory slot" in result

    async def test_edit_memory_multiple_occurrences(self, memory_toolkit):
        """Test editing with multiple occurrences warning."""
        await memory_toolkit.write_memory("test test test", "test_slot")

        result = await memory_toolkit.edit_memory("test", "exam", "test_slot")

        assert isinstance(result, str)
        assert "Found 3 occurrences" in result
        assert "more specific context" in result

    async def test_append_to_memory(self, memory_toolkit):
        """Test appending content to memory."""
        # Write initial content
        await memory_toolkit.write_memory("Initial content", "test_slot")

        # Append content
        result = await memory_toolkit.append_to_memory("Appended content", "test_slot")

        assert isinstance(result, str)
        assert "Successfully appended" in result

        # Verify combined content
        read_result = await memory_toolkit.read_memory("test_slot")
        assert read_result == "Initial content\nAppended content"

    async def test_append_to_empty_slot(self, memory_toolkit):
        """Test appending to empty memory slot."""
        result = await memory_toolkit.append_to_memory("First content", "new_slot")

        assert isinstance(result, str)
        assert "Successfully appended" in result

        read_result = await memory_toolkit.read_memory("new_slot")
        assert read_result == "First content"

    async def test_clear_memory(self, memory_toolkit):
        """Test clearing memory slot."""
        # Write content
        await memory_toolkit.write_memory("Content to clear", "test_slot")

        # Clear memory
        result = await memory_toolkit.clear_memory("test_slot")

        assert isinstance(result, str)
        assert "has been cleared" in result

        # Verify slot is empty
        read_result = await memory_toolkit.read_memory("test_slot")
        assert "empty or does not exist" in read_result

    async def test_list_memory_slots(self, memory_toolkit):
        """Test listing memory slots."""
        # Add some content to different slots
        await memory_toolkit.write_memory("Content 1", "slot1")
        await memory_toolkit.write_memory("Content 2 with more text", "slot2")

        result = await memory_toolkit.list_memory_slots()

        assert isinstance(result, str)
        assert "slot1" in result
        assert "slot2" in result
        assert "9 chars" in result  # Length of "Content 1"

    async def test_list_empty_memory(self, memory_toolkit):
        """Test listing when no memory slots exist."""
        result = await memory_toolkit.list_memory_slots()

        assert isinstance(result, str)
        assert "No memory slots exist" in result

    async def test_search_memory(self, memory_toolkit):
        """Test searching within memory slots."""
        # Add content to search
        await memory_toolkit.write_memory("Python is great\nJava is also good", "languages")
        await memory_toolkit.write_memory("Machine learning with Python", "ml")

        # Search for "Python"
        result = await memory_toolkit.search_memory("Python")

        assert isinstance(result, str)
        assert "languages" in result
        assert "ml" in result
        assert "Python" in result

    async def test_search_memory_no_results(self, memory_toolkit):
        """Test searching with no matches."""
        await memory_toolkit.write_memory("Some content", "test_slot")

        result = await memory_toolkit.search_memory("nonexistent")

        assert isinstance(result, str)
        assert "No matches found" in result

    async def test_search_memory_specific_slot(self, memory_toolkit):
        """Test searching in specific memory slot."""
        await memory_toolkit.write_memory("Python programming", "slot1")
        await memory_toolkit.write_memory("Java development", "slot2")

        result = await memory_toolkit.search_memory("Python", slot_name="slot1")

        assert isinstance(result, str)
        assert "slot1" in result
        assert "slot2" not in result

    async def test_get_memory_stats(self, memory_toolkit):
        """Test getting memory statistics."""
        # Add some content
        await memory_toolkit.write_memory("Short", "slot1")
        await memory_toolkit.write_memory("This is a longer piece of content", "slot2")

        result = await memory_toolkit.get_memory_stats()

        assert isinstance(result, str)
        assert "Total slots: 2" in result
        assert "Total characters:" in result
        assert "Average slot size:" in result
        assert "Largest slot:" in result

    async def test_get_memory_stats_empty(self, memory_toolkit):
        """Test getting stats when no memory exists."""
        result = await memory_toolkit.get_memory_stats()

        assert isinstance(result, str)
        assert "No memory slots exist" in result

    async def test_content_size_limit(self, memory_toolkit):
        """Test content size validation."""
        # Try to write content larger than limit (1000 chars)
        large_content = "x" * 1001

        result = await memory_toolkit.write_memory(large_content, "test_slot")

        assert isinstance(result, str)
        assert "Content too large" in result

    async def test_default_slot_name(self, memory_toolkit):
        """Test using default slot name."""
        await memory_toolkit.write_memory("Default slot content")

        result = await memory_toolkit.read_memory()  # No slot name provided

        assert result == "Default slot content"


class TestFileBasedMemoryToolkit:
    """Test cases for file-based MemoryToolkit."""

    async def test_file_persistence(self, file_memory_toolkit):
        """Test that memory persists to files."""
        # Write content
        await file_memory_toolkit.write_memory("Persistent content", "test_slot")

        # Check that file was created
        storage_dir = Path(file_memory_toolkit.storage_dir)
        assert (storage_dir / "test_slot.txt").exists()

        # Read file content directly
        with open(storage_dir / "test_slot.txt", "r") as f:
            file_content = f.read()
        assert file_content == "Persistent content"

    async def test_load_from_existing_files(self, file_memory_config):
        """Test loading memory from existing files."""
        # Create a file manually
        storage_dir = Path(file_memory_config.config["storage_dir"])
        test_file = storage_dir / "existing_slot.txt"
        with open(test_file, "w") as f:
            f.write("Pre-existing content")

        # Create toolkit (should load existing files)
        toolkit = get_toolkit("memory", file_memory_config)

        # Read the pre-existing content
        result = await toolkit.read_memory("existing_slot")
        assert result == "Pre-existing content"

    async def test_file_cleanup_on_clear(self, file_memory_toolkit):
        """Test that files are removed when memory is cleared."""
        # Write content
        await file_memory_toolkit.write_memory("Content to remove", "test_slot")

        storage_dir = Path(file_memory_toolkit.storage_dir)
        test_file = storage_dir / "test_slot.txt"
        assert test_file.exists()

        # Clear memory
        await file_memory_toolkit.clear_memory("test_slot")

        # File should be removed
        assert not test_file.exists()


class TestMemoryToolkitEdgeCases:
    """Test edge cases and error conditions."""

    async def test_edit_nonexistent_slot(self, memory_toolkit):
        """Test editing non-existent memory slot."""
        result = await memory_toolkit.edit_memory("old", "new", "nonexistent")

        assert isinstance(result, str)
        assert "does not exist" in result

    async def test_filename_sanitization(self, file_memory_toolkit):
        """Test that slot names are sanitized for filenames."""
        # Use slot name with special characters
        slot_name = "test/slot:with*special?chars"
        await file_memory_toolkit.write_memory("Test content", slot_name)

        # Should create a sanitized filename
        storage_dir = Path(file_memory_toolkit.storage_dir)
        files = list(storage_dir.glob("*.txt"))
        assert len(files) == 1

        # Filename should not contain special characters
        filename = files[0].stem
        assert "/" not in filename
        assert ":" not in filename
        assert "*" not in filename
        assert "?" not in filename

    async def test_concurrent_operations(self, memory_toolkit):
        """Test concurrent memory operations."""
        import asyncio

        # Perform multiple operations concurrently
        tasks = [memory_toolkit.write_memory(f"Content {i}", f"slot_{i}") for i in range(5)]

        results = await asyncio.gather(*tasks)

        # All operations should succeed
        for result in results:
            assert "updated successfully" in result

        # Verify all slots exist
        stats = await memory_toolkit.get_memory_stats()
        assert "Total slots: 5" in stats
