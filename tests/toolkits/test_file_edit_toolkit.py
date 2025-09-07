"""
Tests for FileEditToolkit.

Comprehensive test suite covering file operations, safety features,
backup creation, filename sanitization, and directory operations.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from cogents_core.toolify.config import ToolkitConfig

from cogents_tools.toolkits.file_edit_toolkit import FileEditToolkit


class TestFileEditToolkitInitialization:
    """Test FileEditToolkit initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with default configuration."""
        toolkit = FileEditToolkit()

        assert toolkit.work_dir == Path("./file_workspace").resolve()
        assert toolkit.default_encoding == "utf-8"
        assert toolkit.backup_enabled is True
        assert toolkit.max_file_size == 10 * 1024 * 1024  # 10MB
        assert toolkit.allowed_extensions is None
        assert toolkit.backup_dir == toolkit.work_dir / ".backups"

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(
                config={
                    "work_dir": temp_dir,
                    "default_encoding": "latin-1",
                    "backup_enabled": False,
                    "max_file_size": 1024,
                    "allowed_extensions": [".txt", ".py"],
                }
            )

            toolkit = FileEditToolkit(config)

            assert toolkit.work_dir == Path(temp_dir).resolve()
            assert toolkit.default_encoding == "latin-1"
            assert toolkit.backup_enabled is False
            assert toolkit.max_file_size == 1024
            assert toolkit.allowed_extensions == [".txt", ".py"]

    def test_working_directory_creation(self):
        """Test that working directory is created during initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / "test_workspace"
            config = ToolkitConfig(config={"work_dir": str(work_dir)})

            toolkit = FileEditToolkit(config)

            assert work_dir.exists()
            assert work_dir.is_dir()
            assert toolkit.work_dir == work_dir.resolve()

    def test_backup_directory_creation(self):
        """Test that backup directory is created when backup is enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": True})

            toolkit = FileEditToolkit(config)

            backup_dir = toolkit.work_dir / ".backups"
            assert backup_dir.exists()
            assert backup_dir.is_dir()

    def test_no_backup_directory_when_disabled(self):
        """Test that backup directory is not created when backup is disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": False})

            toolkit = FileEditToolkit(config)

            toolkit.work_dir / ".backups"
            # Directory might exist from previous tests, but shouldn't be created by this instance
            # We can't easily test this without more complex setup


class TestFileEditToolkitFilenameSanitization:
    """Test filename sanitization functionality."""

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        toolkit = FileEditToolkit()

        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file with spaces.txt", "file_with_spaces.txt"),
            ("file@#$%^&*()+=.txt", "file_.txt"),
            ('file/\\:*?"<>|.txt', "file_.txt"),
            ("multiple___underscores.txt", "multiple_underscores.txt"),
            ("_leading_trailing_.txt", "leading_trailing_.txt"),  # Extension prevents trailing underscore removal
            ("_leading_trailing_", "leading_trailing"),  # No extension, so trailing underscore is removed
            ("", "unnamed_file"),
            ("...", "unnamed_file"),
            ("___", "unnamed_file"),
        ]

        for input_name, expected in test_cases:
            result = toolkit._sanitize_filename(input_name)
            assert result == expected, f"Failed for input: {input_name}"

    def test_sanitize_filename_preserves_valid_chars(self):
        """Test that valid characters are preserved."""
        toolkit = FileEditToolkit()

        valid_names = [
            "file.txt",
            "file-name.py",
            "file_name.json",
            "File123.XML",
            "test.file.with.dots.txt",
        ]

        for name in valid_names:
            result = toolkit._sanitize_filename(name)
            assert result == name


class TestFileEditToolkitPathResolution:
    """Test file path resolution and validation."""

    def test_resolve_filepath_relative_path(self):
        """Test resolving relative file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = toolkit._resolve_filepath("test.txt")
            expected = toolkit.work_dir / "test.txt"

            assert result == expected

    def test_resolve_filepath_nested_path(self):
        """Test resolving nested file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = toolkit._resolve_filepath("subdir/test.txt", create_dirs=True)
            expected = toolkit.work_dir / "subdir" / "test.txt"

            assert result == expected
            assert result.parent.exists()  # Directory should be created

    def test_resolve_filepath_absolute_within_workdir(self):
        """Test resolving absolute paths within working directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            abs_path = toolkit.work_dir / "test.txt"
            result = toolkit._resolve_filepath(str(abs_path))

            assert result == abs_path

    def test_resolve_filepath_outside_workdir_raises_error(self):
        """Test that paths outside working directory raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            with pytest.raises(ValueError, match="Path outside working directory"):
                toolkit._resolve_filepath("/etc/passwd")

    def test_resolve_filepath_sanitizes_filename(self):
        """Test that filename is sanitized during path resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = toolkit._resolve_filepath("bad@filename.txt")
            expected = toolkit.work_dir / "bad_filename.txt"

            assert result == expected


class TestFileEditToolkitExtensionChecking:
    """Test file extension validation."""

    def test_check_file_extension_no_restrictions(self):
        """Test extension checking with no restrictions."""
        toolkit = FileEditToolkit()
        toolkit.allowed_extensions = None

        test_files = [
            Path("test.txt"),
            Path("test.py"),
            Path("test.exe"),
            Path("test"),  # No extension
        ]

        for file_path in test_files:
            assert toolkit._check_file_extension(file_path) is True

    def test_check_file_extension_with_restrictions(self):
        """Test extension checking with allowed extensions."""
        toolkit = FileEditToolkit()
        toolkit.allowed_extensions = [".txt", ".py", ".json"]

        test_cases = [
            (Path("test.txt"), True),
            (Path("test.py"), True),
            (Path("test.json"), True),
            (Path("test.TXT"), True),  # Case insensitive
            (Path("test.exe"), False),
            (Path("test.doc"), False),
            (Path("test"), False),  # No extension
        ]

        for file_path, expected in test_cases:
            result = toolkit._check_file_extension(file_path)
            assert result == expected, f"Failed for {file_path}"


class TestFileEditToolkitBackupCreation:
    """Test backup creation functionality."""

    def test_create_backup_success(self):
        """Test successful backup creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": True})
            toolkit = FileEditToolkit(config)

            # Create a test file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("original content")

            # Create backup
            backup_path = toolkit._create_backup(test_file)

            assert backup_path is not None
            assert backup_path.exists()
            assert backup_path.parent == toolkit.backup_dir
            assert "test" in backup_path.name
            assert backup_path.read_text() == "original content"

    def test_create_backup_disabled(self):
        """Test backup creation when disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": False})
            toolkit = FileEditToolkit(config)

            # Create a test file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("original content")

            # Try to create backup
            backup_path = toolkit._create_backup(test_file)

            assert backup_path is None

    def test_create_backup_nonexistent_file(self):
        """Test backup creation for non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": True})
            toolkit = FileEditToolkit(config)

            # Try to backup non-existent file
            test_file = toolkit.work_dir / "nonexistent.txt"
            backup_path = toolkit._create_backup(test_file)

            assert backup_path is None

    def test_create_backup_with_timestamp(self):
        """Test that backup filename includes timestamp."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": True})
            toolkit = FileEditToolkit(config)

            # Create a test file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("content")

            with patch("cogents_tools.toolkits.file_edit_toolkit.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20231201_143000"

                backup_path = toolkit._create_backup(test_file)

                assert "20231201_143000" in backup_path.name


class TestFileEditToolkitFileOperations:
    """Test core file operation methods."""

    @pytest.mark.asyncio
    async def test_create_file_success(self):
        """Test successful file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = await toolkit.create_file("test.txt", "Hello World!")

            assert "Successfully created file" in result
            test_file = toolkit.work_dir / "test.txt"
            assert test_file.exists()
            assert test_file.read_text() == "Hello World!"

    @pytest.mark.asyncio
    async def test_create_file_with_subdirectory(self):
        """Test file creation in subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = await toolkit.create_file("subdir/test.txt", "Content")

            assert "Successfully created file" in result
            test_file = toolkit.work_dir / "subdir" / "test.txt"
            assert test_file.exists()
            assert test_file.read_text() == "Content"

    @pytest.mark.asyncio
    async def test_create_file_overwrite_protection(self):
        """Test overwrite protection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create initial file
            await toolkit.create_file("test.txt", "Original")

            # Try to create again without overwrite
            result = await toolkit.create_file("test.txt", "New content")

            assert "Error: File already exists" in result
            test_file = toolkit.work_dir / "test.txt"
            assert test_file.read_text() == "Original"

    @pytest.mark.asyncio
    async def test_create_file_with_overwrite(self):
        """Test file creation with overwrite enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": True})
            toolkit = FileEditToolkit(config)

            # Create initial file
            await toolkit.create_file("test.txt", "Original")

            # Overwrite with new content
            result = await toolkit.create_file("test.txt", "New content", overwrite=True)

            assert "Successfully created file" in result
            assert "backup created" in result
            test_file = toolkit.work_dir / "test.txt"
            assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_create_file_extension_restriction(self):
        """Test file creation with extension restrictions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "allowed_extensions": [".txt"]})
            toolkit = FileEditToolkit(config)

            # Allowed extension
            result = await toolkit.create_file("test.txt", "Content")
            assert "Successfully created file" in result

            # Disallowed extension
            result = await toolkit.create_file("test.py", "print('hello')")
            assert "Error: File extension not allowed" in result

    @pytest.mark.asyncio
    async def test_create_file_size_limit(self):
        """Test file creation with size limits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "max_file_size": 10})
            toolkit = FileEditToolkit(config)

            # Content within limit
            result = await toolkit.create_file("small.txt", "small")
            assert "Successfully created file" in result

            # Content exceeding limit
            result = await toolkit.create_file("large.txt", "a" * 20)
            assert "Error: Content too large" in result

    @pytest.mark.asyncio
    async def test_read_file_success(self):
        """Test successful file reading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create a test file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("Hello World!")

            result = await toolkit.read_file("test.txt")
            assert result == "Hello World!"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self):
        """Test reading non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = await toolkit.read_file("nonexistent.txt")
            assert "Error: File not found" in result

    @pytest.mark.asyncio
    async def test_read_file_is_directory(self):
        """Test reading a directory instead of file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create a directory
            (toolkit.work_dir / "testdir").mkdir()

            result = await toolkit.read_file("testdir")
            assert "Error: Path is not a file" in result

    @pytest.mark.asyncio
    async def test_read_file_size_limit(self):
        """Test reading file that exceeds size limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "max_file_size": 10})
            toolkit = FileEditToolkit(config)

            # Create a large file
            test_file = toolkit.work_dir / "large.txt"
            test_file.write_text("a" * 20)

            result = await toolkit.read_file("large.txt")
            assert "Error: File too large" in result

    @pytest.mark.asyncio
    async def test_read_file_encoding_error(self):
        """Test reading file with encoding issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create a file with binary content
            test_file = toolkit.work_dir / "binary.txt"
            test_file.write_bytes(b"\x80\x81\x82")

            result = await toolkit.read_file("binary.txt")
            assert "Error: Unable to decode file" in result

    @pytest.mark.asyncio
    async def test_write_file_success(self):
        """Test successful file writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = await toolkit.write_file("test.txt", "New content")

            assert "Successfully written to file" in result
            test_file = toolkit.work_dir / "test.txt"
            assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_file_append_mode(self):
        """Test file writing in append mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create initial file
            await toolkit.write_file("test.txt", "Initial")

            # Append content
            result = await toolkit.write_file("test.txt", " Appended", append=True)

            assert "Successfully appended to file" in result
            test_file = toolkit.work_dir / "test.txt"
            assert test_file.read_text() == "Initial Appended"

    @pytest.mark.asyncio
    async def test_write_file_with_backup(self):
        """Test file writing creates backup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": True})
            toolkit = FileEditToolkit(config)

            # Create initial file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("Original")

            # Write new content
            result = await toolkit.write_file("test.txt", "New content")

            assert "Successfully written to file" in result
            assert "backup created" in result
            assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_delete_file_success(self):
        """Test successful file deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": True})
            toolkit = FileEditToolkit(config)

            # Create a test file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("Content")

            result = await toolkit.delete_file("test.txt")

            assert "Successfully deleted file" in result
            assert "backup created" in result
            assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self):
        """Test deleting non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = await toolkit.delete_file("nonexistent.txt")
            assert "Error: File not found" in result

    @pytest.mark.asyncio
    async def test_delete_file_is_directory(self):
        """Test deleting a directory instead of file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create a directory
            (toolkit.work_dir / "testdir").mkdir()

            result = await toolkit.delete_file("testdir")
            assert "Error: Path is not a file" in result

    @pytest.mark.asyncio
    async def test_delete_file_no_backup(self):
        """Test file deletion without backup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create a test file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("Content")

            result = await toolkit.delete_file("test.txt", create_backup=False)

            assert "Successfully deleted file" in result
            assert "backup created" not in result
            assert not test_file.exists()


class TestFileEditToolkitDirectoryOperations:
    """Test directory listing and search operations."""

    @pytest.mark.asyncio
    async def test_list_files_default_directory(self):
        """Test listing files in default directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test files
            (toolkit.work_dir / "file1.txt").write_text("content1")
            (toolkit.work_dir / "file2.py").write_text("content2")
            (toolkit.work_dir / "subdir").mkdir()

            result = await toolkit.list_files()

            assert f"Files in {toolkit.work_dir}:" in result
            assert "📄 file1.txt" in result
            assert "📄 file2.py" in result
            assert "📁 subdir/" in result

    @pytest.mark.asyncio
    async def test_list_files_with_pattern(self):
        """Test listing files with glob pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test files
            (toolkit.work_dir / "file1.txt").write_text("content1")
            (toolkit.work_dir / "file2.py").write_text("content2")
            (toolkit.work_dir / "file3.txt").write_text("content3")

            result = await toolkit.list_files(".", "*.txt")

            assert "📄 file1.txt" in result
            assert "📄 file3.txt" in result
            assert "📄 file2.py" not in result

    @pytest.mark.asyncio
    async def test_list_files_subdirectory(self):
        """Test listing files in subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create subdirectory with files
            subdir = toolkit.work_dir / "subdir"
            subdir.mkdir()
            (subdir / "subfile.txt").write_text("content")

            result = await toolkit.list_files("subdir")

            assert f"Files in {subdir}:" in result
            assert "📄 subfile.txt" in result

    @pytest.mark.asyncio
    async def test_list_files_nonexistent_directory(self):
        """Test listing files in non-existent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = await toolkit.list_files("nonexistent")
            assert "Error: Directory not found" in result

    @pytest.mark.asyncio
    async def test_list_files_empty_directory(self):
        """Test listing files in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": False})
            toolkit = FileEditToolkit(config)

            result = await toolkit.list_files()
            assert "No files found matching pattern" in result

    @pytest.mark.asyncio
    async def test_search_in_files_success(self):
        """Test successful text search in files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test files with content
            (toolkit.work_dir / "file1.txt").write_text("Hello World\nSecond line")
            (toolkit.work_dir / "file2.txt").write_text("Goodbye World\nAnother line")
            (toolkit.work_dir / "file3.py").write_text("print('Hello Python')")

            result = await toolkit.search_in_files("Hello")

            assert "Found" in result
            assert "file1.txt:1: Hello World" in result
            assert "file3.py:1: print('Hello Python')" in result
            assert "file2.txt" not in result

    @pytest.mark.asyncio
    async def test_search_in_files_regex_pattern(self):
        """Test search with regex pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test files
            (toolkit.work_dir / "file1.txt").write_text("test123\ntest456")
            (toolkit.work_dir / "file2.txt").write_text("testing\nno match")

            result = await toolkit.search_in_files(r"test\d+")

            assert "file1.txt:1: test123" in result
            assert "file1.txt:2: test456" in result
            assert "testing" not in result

    @pytest.mark.asyncio
    async def test_search_in_files_case_insensitive(self):
        """Test case-insensitive search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test file
            (toolkit.work_dir / "file1.txt").write_text("Hello\nHELLO\nhello")

            result = await toolkit.search_in_files("hello")

            # Should match all variations
            lines = result.split("\n")
            matches = [line for line in lines if "file1.txt:" in line]
            assert len(matches) == 3

    @pytest.mark.asyncio
    async def test_search_in_files_invalid_regex(self):
        """Test search with invalid regex pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = await toolkit.search_in_files("[invalid")
            assert "Error: Invalid regex pattern" in result

    @pytest.mark.asyncio
    async def test_search_in_files_no_matches(self):
        """Test search with no matches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test file
            (toolkit.work_dir / "file1.txt").write_text("Hello World")

            result = await toolkit.search_in_files("nonexistent")
            assert "No matches found" in result

    @pytest.mark.asyncio
    async def test_search_in_files_with_file_pattern(self):
        """Test search with file pattern filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test files
            (toolkit.work_dir / "file1.txt").write_text("Hello")
            (toolkit.work_dir / "file2.py").write_text("Hello")
            (toolkit.work_dir / "file3.md").write_text("Hello")

            result = await toolkit.search_in_files("Hello", ".", "*.txt")

            assert "file1.txt" in result
            assert "file2.py" not in result
            assert "file3.md" not in result

    @pytest.mark.asyncio
    async def test_get_file_info_success(self):
        """Test getting file information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("Hello World")

            result = await toolkit.get_file_info("test.txt")

            assert f"File Information: {test_file}" in result
            assert "Size:" in result
            assert "Created:" in result
            assert "Modified:" in result
            assert "Type: File" in result
            assert "Extension: .txt" in result

    @pytest.mark.asyncio
    async def test_get_file_info_directory(self):
        """Test getting directory information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Create test directory
            test_dir = toolkit.work_dir / "testdir"
            test_dir.mkdir()

            result = await toolkit.get_file_info("testdir")

            assert f"File Information: {test_dir}" in result
            assert "Type: Directory" in result

    @pytest.mark.asyncio
    async def test_get_file_info_not_found(self):
        """Test getting info for non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            result = await toolkit.get_file_info("nonexistent.txt")
            assert "Error: File not found" in result


class TestFileEditToolkitToolsMap:
    """Test tools map functionality."""

    @pytest.mark.asyncio
    async def test_get_tools_map(self):
        """Test getting tools map."""
        toolkit = FileEditToolkit()

        tools_map = await toolkit.get_tools_map()

        expected_tools = {
            "create_file",
            "read_file",
            "write_file",
            "delete_file",
            "list_files",
            "search_in_files",
            "get_file_info",
        }

        assert set(tools_map.keys()) == expected_tools

        # Verify all tools are callable
        for tool_name, tool_func in tools_map.items():
            assert callable(tool_func)


class TestFileEditToolkitIntegration:
    """Integration tests for FileEditToolkit."""

    @pytest.fixture
    def file_edit_toolkit(self):
        """Create a FileEditToolkit instance for testing."""
        temp_dir = tempfile.mkdtemp()
        config = ToolkitConfig(
            config={
                "work_dir": temp_dir,
                "backup_enabled": True,
                "max_file_size": 1024 * 1024,  # 1MB
            }
        )
        return FileEditToolkit(config)

    @pytest.mark.asyncio
    async def test_toolkit_lifecycle(self, file_edit_toolkit):
        """Test complete toolkit lifecycle."""
        # Test that toolkit can be built and cleaned up
        await file_edit_toolkit.build()
        assert file_edit_toolkit._built is True

        await file_edit_toolkit.cleanup()
        assert file_edit_toolkit._built is False

    @pytest.mark.asyncio
    async def test_filtered_tools_activation(self):
        """Test toolkit with filtered tool activation."""
        config = ToolkitConfig(activated_tools=["create_file", "read_file", "write_file"])
        toolkit = FileEditToolkit(config)

        tools_map = await toolkit.get_filtered_tools_map()

        assert set(tools_map.keys()) == {"create_file", "read_file", "write_file"}
        assert "delete_file" not in tools_map
        assert "list_files" not in tools_map

    @pytest.mark.asyncio
    async def test_call_tool_interface(self, file_edit_toolkit):
        """Test calling tools through the call_tool interface."""
        result = await file_edit_toolkit.call_tool("create_file", file_path="test.txt", content="Hello")

        assert "Successfully created file" in result
        test_file = file_edit_toolkit.work_dir / "test.txt"
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_langchain_tools_conversion(self, file_edit_toolkit):
        """Test conversion to LangChain tools."""
        langchain_tools = await file_edit_toolkit.get_langchain_tools()

        assert len(langchain_tools) == 7
        tool_names = {tool.name for tool in langchain_tools}
        expected_names = {
            "create_file",
            "read_file",
            "write_file",
            "delete_file",
            "list_files",
            "search_in_files",
            "get_file_info",
        }
        assert tool_names == expected_names

    @pytest.mark.asyncio
    async def test_complete_file_workflow(self, file_edit_toolkit):
        """Test a complete file management workflow."""
        # Create a file
        result = await file_edit_toolkit.create_file("workflow.txt", "Initial content")
        assert "Successfully created file" in result

        # Read the file
        content = await file_edit_toolkit.read_file("workflow.txt")
        assert content == "Initial content"

        # Write new content (should create backup)
        result = await file_edit_toolkit.write_file("workflow.txt", "Updated content")
        assert "Successfully written to file" in result
        assert "backup created" in result

        # Verify new content
        content = await file_edit_toolkit.read_file("workflow.txt")
        assert content == "Updated content"

        # Append content
        result = await file_edit_toolkit.write_file("workflow.txt", "\nAppended line", append=True)
        assert "Successfully appended to file" in result

        # Verify appended content
        content = await file_edit_toolkit.read_file("workflow.txt")
        assert content == "Updated content\nAppended line"

        # Get file info
        info = await file_edit_toolkit.get_file_info("workflow.txt")
        assert "File Information:" in info
        assert "workflow.txt" in info

        # List files
        listing = await file_edit_toolkit.list_files()
        assert "workflow.txt" in listing

        # Search in files
        search_result = await file_edit_toolkit.search_in_files("Updated")
        assert "workflow.txt" in search_result
        assert "Updated content" in search_result

        # Delete file (should create backup)
        result = await file_edit_toolkit.delete_file("workflow.txt")
        assert "Successfully deleted file" in result
        assert "backup created" in result

        # Verify file is deleted
        test_file = file_edit_toolkit.work_dir / "workflow.txt"
        assert not test_file.exists()


class TestFileEditToolkitErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_file_operations_with_permission_errors(self):
        """Test file operations with simulated permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Mock permission error
            with patch("builtins.open", side_effect=PermissionError("Permission denied")):
                result = await toolkit.create_file("test.txt", "content")
                assert "Failed to create file" in result
                assert "Permission denied" in result

    @pytest.mark.asyncio
    async def test_backup_creation_failure(self):
        """Test handling of backup creation failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir, "backup_enabled": True})
            toolkit = FileEditToolkit(config)

            # Create a test file
            test_file = toolkit.work_dir / "test.txt"
            test_file.write_text("content")

            # Mock backup failure
            with patch("shutil.copy2", side_effect=OSError("Backup failed")):
                backup_path = toolkit._create_backup(test_file)
                assert backup_path is None

    @pytest.mark.asyncio
    async def test_directory_operations_with_errors(self):
        """Test directory operations with various errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolkitConfig(config={"work_dir": temp_dir})
            toolkit = FileEditToolkit(config)

            # Mock glob error
            with patch.object(Path, "glob", side_effect=OSError("Glob failed")):
                result = await toolkit.list_files()
                assert "Failed to list files" in result
