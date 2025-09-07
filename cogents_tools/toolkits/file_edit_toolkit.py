"""
File editing toolkit for file operations and content management.

Provides tools for creating, reading, writing, and managing files with
safety features, backup capabilities, and comprehensive error handling.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


@register_toolkit("file_edit")
class FileEditToolkit(AsyncBaseToolkit):
    """
    Toolkit for file operations and content management.

    This toolkit provides comprehensive file management capabilities including:
    - File creation, reading, and writing
    - Directory operations
    - File backup and versioning
    - Safe filename handling
    - Content search and replacement
    - File metadata operations

    Features:
    - Automatic filename sanitization
    - Backup creation before modifications
    - Configurable working directory
    - Multiple encoding support
    - Path resolution and validation
    - Comprehensive error handling

    Safety features:
    - Prevents overwriting without backup
    - Validates file paths and permissions
    - Sanitizes filenames to prevent security issues
    - Configurable file size limits
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the file edit toolkit.

        Args:
            config: Toolkit configuration containing directory and settings
        """
        super().__init__(config)

        # Configuration
        self.work_dir = Path(self.config.config.get("work_dir", "./file_workspace")).resolve()
        self.default_encoding = self.config.config.get("default_encoding", "utf-8")
        self.backup_enabled = self.config.config.get("backup_enabled", True)
        self.max_file_size = self.config.config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        self.allowed_extensions = self.config.config.get("allowed_extensions", None)  # None = all allowed

        # Create working directory
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Backup directory
        self.backup_dir = self.work_dir / ".backups"
        if self.backup_enabled:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"FileEditToolkit initialized with work directory: {self.work_dir}")

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename by replacing unsafe characters.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for filesystem use
        """
        # Replace unsafe characters with underscores
        safe = re.sub(r"[^\w\-_.]", "_", filename)

        # Remove multiple consecutive underscores
        safe = re.sub(r"_+", "_", safe)

        # Remove leading/trailing underscores and dots
        safe = safe.strip("_.")

        # Ensure filename is not empty
        if not safe:
            safe = "unnamed_file"

        return safe

    def _resolve_filepath(self, file_path: str, create_dirs: bool = False) -> Path:
        """
        Resolve and validate a file path within the working directory.

        Args:
            file_path: File path to resolve
            create_dirs: Whether to create parent directories

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or outside working directory
        """
        # Convert to Path object
        path = Path(file_path)

        # If not absolute, make it relative to work_dir
        if not path.is_absolute():
            path = self.work_dir / path

        # Resolve to absolute path
        path = path.resolve()

        # Security check: ensure path is within work_dir
        try:
            path.relative_to(self.work_dir)
        except ValueError:
            raise ValueError(f"Path outside working directory: {path}")

        # Sanitize filename
        if path.name:
            sanitized_name = self._sanitize_filename(path.name)
            path = path.parent / sanitized_name

        # Create parent directories if requested
        if create_dirs and path.parent != path:
            path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def _check_file_extension(self, file_path: Path) -> bool:
        """
        Check if file extension is allowed.

        Args:
            file_path: Path to check

        Returns:
            True if extension is allowed
        """
        if self.allowed_extensions is None:
            return True

        extension = file_path.suffix.lower()
        return extension in [ext.lower() for ext in self.allowed_extensions]

    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create a backup of an existing file.

        Args:
            file_path: Path to file to backup

        Returns:
            Path to backup file, or None if backup failed
        """
        if not self.backup_enabled or not file_path.exists():
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name

            shutil.copy2(file_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
            return None

    async def create_file(
        self, file_path: str, content: str = "", encoding: Optional[str] = None, overwrite: bool = False
    ) -> str:
        """
        Create a new file with the specified content.

        This tool creates a new file in the working directory with the given content.
        It includes safety features like filename sanitization, backup creation,
        and overwrite protection.

        Args:
            file_path: Path for the new file (relative to working directory)
            content: Initial content for the file (default: empty string)
            encoding: Text encoding to use (default: utf-8)
            overwrite: Whether to overwrite existing files (default: False)

        Returns:
            Success message with file path or error description

        Examples:
            - create_file("notes.txt", "Hello World!")
            - create_file("data/config.json", '{"key": "value"}')
            - create_file("script.py", "print('Hello')", overwrite=True)
        """
        encoding = encoding or self.default_encoding

        try:
            # Resolve and validate path
            resolved_path = self._resolve_filepath(file_path, create_dirs=True)

            # Check file extension
            if not self._check_file_extension(resolved_path):
                return f"Error: File extension not allowed: {resolved_path.suffix}"

            # Check if file exists and handle overwrite
            if resolved_path.exists() and not overwrite:
                return f"Error: File already exists: {resolved_path}. Use overwrite=True to replace."

            # Create backup if file exists
            backup_path = self._create_backup(resolved_path)

            # Check content size
            content_size = len(content.encode(encoding))
            if content_size > self.max_file_size:
                return f"Error: Content too large ({content_size} bytes, max: {self.max_file_size})"

            # Write the file
            with open(resolved_path, "w", encoding=encoding) as f:
                f.write(content)

            result_msg = f"Successfully created file: {resolved_path}"
            if backup_path:
                result_msg += f" (backup created: {backup_path.name})"

            self.logger.info(result_msg)
            return result_msg

        except Exception as e:
            error_msg = f"Failed to create file '{file_path}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def read_file(self, file_path: str, encoding: Optional[str] = None) -> str:
        """
        Read the contents of a file.

        Args:
            file_path: Path to the file to read
            encoding: Text encoding to use (default: utf-8)

        Returns:
            File contents or error message
        """
        encoding = encoding or self.default_encoding

        try:
            resolved_path = self._resolve_filepath(file_path)

            if not resolved_path.exists():
                return f"Error: File not found: {resolved_path}"

            if not resolved_path.is_file():
                return f"Error: Path is not a file: {resolved_path}"

            # Check file size
            file_size = resolved_path.stat().st_size
            if file_size > self.max_file_size:
                return f"Error: File too large ({file_size} bytes, max: {self.max_file_size})"

            with open(resolved_path, "r", encoding=encoding) as f:
                content = f.read()

            self.logger.info(f"Read file: {resolved_path} ({len(content)} characters)")
            return content

        except UnicodeDecodeError as e:
            return f"Error: Unable to decode file with {encoding} encoding: {str(e)}"
        except Exception as e:
            error_msg = f"Failed to read file '{file_path}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def write_file(
        self, file_path: str, content: str, encoding: Optional[str] = None, append: bool = False
    ) -> str:
        """
        Write content to a file.

        This tool writes content to an existing file or creates a new one.
        It automatically creates backups of existing files before modification.

        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            encoding: Text encoding to use (default: utf-8)
            append: Whether to append to existing content (default: False)

        Returns:
            Success message or error description
        """
        encoding = encoding or self.default_encoding

        try:
            resolved_path = self._resolve_filepath(file_path, create_dirs=True)

            # Check file extension
            if not self._check_file_extension(resolved_path):
                return f"Error: File extension not allowed: {resolved_path.suffix}"

            # Create backup if file exists
            backup_path = self._create_backup(resolved_path)

            # Check content size
            if append and resolved_path.exists():
                existing_size = resolved_path.stat().st_size
                new_content_size = len(content.encode(encoding))
                total_size = existing_size + new_content_size
            else:
                total_size = len(content.encode(encoding))

            if total_size > self.max_file_size:
                return f"Error: Total content too large ({total_size} bytes, max: {self.max_file_size})"

            # Write the file
            mode = "a" if append else "w"
            with open(resolved_path, mode, encoding=encoding) as f:
                f.write(content)

            action = "appended to" if append else "written to"
            result_msg = f"Successfully {action} file: {resolved_path}"
            if backup_path:
                result_msg += f" (backup created: {backup_path.name})"

            self.logger.info(result_msg)
            return result_msg

        except Exception as e:
            error_msg = f"Failed to write file '{file_path}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def delete_file(self, file_path: str, create_backup: bool = True) -> str:
        """
        Delete a file.

        Args:
            file_path: Path to the file to delete
            create_backup: Whether to create a backup before deletion

        Returns:
            Success message or error description
        """
        try:
            resolved_path = self._resolve_filepath(file_path)

            if not resolved_path.exists():
                return f"Error: File not found: {resolved_path}"

            if not resolved_path.is_file():
                return f"Error: Path is not a file: {resolved_path}"

            # Create backup if requested
            backup_path = None
            if create_backup:
                backup_path = self._create_backup(resolved_path)

            # Delete the file
            resolved_path.unlink()

            result_msg = f"Successfully deleted file: {resolved_path}"
            if backup_path:
                result_msg += f" (backup created: {backup_path.name})"

            self.logger.info(result_msg)
            return result_msg

        except Exception as e:
            error_msg = f"Failed to delete file '{file_path}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def list_files(self, directory: str = ".", pattern: str = "*") -> str:
        """
        List files in a directory.

        Args:
            directory: Directory to list (relative to working directory)
            pattern: Glob pattern to filter files (default: "*" for all files)

        Returns:
            Formatted list of files and directories
        """
        try:
            if directory == ".":
                dir_path = self.work_dir
            else:
                dir_path = self._resolve_filepath(directory)

            if not dir_path.exists():
                return f"Error: Directory not found: {dir_path}"

            if not dir_path.is_dir():
                return f"Error: Path is not a directory: {dir_path}"

            # Get matching files
            files = list(dir_path.glob(pattern))
            files.sort()

            if not files:
                return f"No files found matching pattern '{pattern}' in {dir_path}"

            # Format the listing
            result_lines = [f"Files in {dir_path}:"]

            for file_path in files:
                if file_path.is_file():
                    size = file_path.stat().st_size
                    modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    result_lines.append(f"  📄 {file_path.name} ({size} bytes, {modified.strftime('%Y-%m-%d %H:%M')})")
                elif file_path.is_dir():
                    result_lines.append(f"  📁 {file_path.name}/")

            return "\n".join(result_lines)

        except Exception as e:
            error_msg = f"Failed to list files in '{directory}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def search_in_files(self, pattern: str, directory: str = ".", file_pattern: str = "*") -> str:
        """
        Search for text patterns within files.

        Args:
            pattern: Text pattern to search for (supports regex)
            directory: Directory to search in (default: current)
            file_pattern: File pattern to include in search (default: all files)

        Returns:
            Search results with file names and line numbers
        """
        try:
            if directory == ".":
                dir_path = self.work_dir
            else:
                dir_path = self._resolve_filepath(directory)

            if not dir_path.exists() or not dir_path.is_dir():
                return f"Error: Invalid directory: {dir_path}"

            # Compile regex pattern
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"

            # Search files
            results = []
            files_searched = 0

            for file_path in dir_path.rglob(file_pattern):
                if not file_path.is_file():
                    continue

                # Skip binary files and backups
                if file_path.suffix.lower() in [".exe", ".bin", ".jpg", ".png", ".pdf"]:
                    continue
                if file_path.parent.name == ".backups":
                    continue

                try:
                    with open(file_path, "r", encoding=self.default_encoding, errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"{file_path.name}:{line_num}: {line.strip()}")

                    files_searched += 1

                except Exception:
                    continue  # Skip files that can't be read

            if not results:
                return f"No matches found for pattern '{pattern}' in {files_searched} files"

            result_text = f"Found {len(results)} matches in {files_searched} files:\n\n"
            result_text += "\n".join(results[:50])  # Limit to first 50 results

            if len(results) > 50:
                result_text += f"\n\n... and {len(results) - 50} more matches"

            return result_text

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def get_file_info(self, file_path: str) -> str:
        """
        Get detailed information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Formatted file information
        """
        try:
            resolved_path = self._resolve_filepath(file_path)

            if not resolved_path.exists():
                return f"Error: File not found: {resolved_path}"

            stat = resolved_path.stat()

            info_lines = [
                f"File Information: {resolved_path}",
                f"Size: {stat.st_size} bytes ({stat.st_size / 1024:.1f} KB)",
                f"Created: {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
                f"Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
                f"Accessed: {datetime.fromtimestamp(stat.st_atime).strftime('%Y-%m-%d %H:%M:%S')}",
                f"Type: {'File' if resolved_path.is_file() else 'Directory'}",
                f"Extension: {resolved_path.suffix or 'None'}",
                f"Permissions: {oct(stat.st_mode)[-3:]}",
            ]

            return "\n".join(info_lines)

        except Exception as e:
            error_msg = f"Failed to get file info for '{file_path}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "create_file": self.create_file,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "delete_file": self.delete_file,
            "list_files": self.list_files,
            "search_in_files": self.search_in_files,
            "get_file_info": self.get_file_info,
        }
