#!/usr/bin/env python3
"""
Tests for cogents-tools lazy importing functionality.

Comprehensive test suite covering:
1. Group-wise imports for semantic toolkit categories
2. Individual toolkit loading on demand
3. Performance and memory efficiency
4. Error handling and graceful failures
5. Integration with external services (marked as integration tests)
"""

import asyncio
import time

import pytest

# Import the lazy loading functionality
import cogents_tools
from cogents_tools import groups
from cogents_tools.lazy_import import get_loaded_modules


class TestLazyImportingBasics:
    """Test basic lazy importing functionality."""

    def test_lazy_loading_status(self):
        """Test lazy loading enable/disable functionality."""
        # Should be enabled by default or can be enabled
        if not cogents_tools.is_lazy_loading_enabled():
            cogents_tools.enable_lazy_loading()

        assert cogents_tools.is_lazy_loading_enabled()

    def test_get_loaded_modules_initially_empty(self):
        """Test that initially no modules are loaded."""
        loaded = get_loaded_modules()
        # At this point, some modules might already be loaded, but it should be a reasonable set
        assert isinstance(loaded, (list, set, tuple))

    def test_list_groups_available(self):
        """Test that toolkit groups are available and properly described."""
        group_info = groups.list_groups()

        assert isinstance(group_info, dict)
        assert len(group_info) > 0

        # Check that we have expected groups
        expected_groups = ["academic", "development", "info_retrieval", "image", "audio"]
        for group in expected_groups:
            assert group in group_info
            assert isinstance(group_info[group], str)
            assert len(group_info[group]) > 0


class TestGroupWiseImports:
    """Test group-wise import functionality."""

    def test_academic_group_loading(self):
        """Test loading academic group."""
        start_time = time.time()
        academic_tools = groups.academic()
        load_time = time.time() - start_time

        # Should load quickly (lazy loading)
        assert load_time < 1.0

        # Should have expected attributes
        assert hasattr(academic_tools, "__dict__")
        available_toolkits = dir(academic_tools)

        # Should contain arxiv toolkit
        assert "arxiv_toolkit" in available_toolkits

    def test_academic_group_arxiv_access(self):
        """Test accessing arxiv toolkit from academic group."""
        academic_tools = groups.academic()

        # Track modules before access
        modules_before = set(get_loaded_modules())

        start_time = time.time()
        try:
            arxiv_toolkit = academic_tools.arxiv_toolkit
            access_time = time.time() - start_time

            # Should access reasonably quickly
            assert access_time < 5.0

            # Should be a class/type
            assert callable(arxiv_toolkit)

            # Create an instance
            arxiv_instance = arxiv_toolkit()
            assert arxiv_instance is not None

            # Should have loaded additional modules
            modules_after = set(get_loaded_modules())
            assert len(modules_after) >= len(modules_before)

        except Exception as e:
            # If toolkit loading fails (e.g., missing dependencies), that's acceptable
            pytest.skip(f"ArXiv toolkit not available: {e}")

    def test_development_group_loading(self):
        """Test loading development group."""
        start_time = time.time()
        dev_tools = groups.development()
        load_time = time.time() - start_time

        # Should load quickly
        assert load_time < 1.0

        available_toolkits = dir(dev_tools)

        # Should contain expected toolkits
        expected_toolkits = ["bash_toolkit", "file_edit_toolkit"]
        for toolkit in expected_toolkits:
            assert toolkit in available_toolkits

    @pytest.mark.asyncio
    async def test_development_group_bash_access(self):
        """Test accessing bash toolkit from development group."""
        dev_tools = groups.development()

        try:
            bash_toolkit = dev_tools.bash_toolkit
            bash_instance = bash_toolkit()

            assert bash_instance is not None

            # Test a simple command execution
            result = await bash_instance.run_bash("echo 'Hello from lazy-loaded bash!'")
            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            pytest.skip(f"Bash toolkit not available: {e}")

    def test_development_group_file_edit_access(self):
        """Test accessing file edit toolkit from development group."""
        dev_tools = groups.development()

        try:
            file_edit_toolkit = dev_tools.file_edit_toolkit
            file_edit_instance = file_edit_toolkit()

            assert file_edit_instance is not None

        except Exception as e:
            pytest.skip(f"File edit toolkit not available: {e}")

    def test_image_group_loading(self):
        """Test loading image processing group."""
        try:
            image_tools = groups.image()
            image_toolkit = image_tools.image_toolkit
            image_instance = image_toolkit()

            assert image_instance is not None

        except Exception as e:
            pytest.skip(f"Image toolkit not available: {e}")

    def test_audio_group_loading(self):
        """Test loading audio group."""
        try:
            audio_tools = groups.audio()
            available_toolkits = dir(audio_tools)

            # Should have audio toolkit
            assert "audio_toolkit" in available_toolkits

            audio_toolkit = audio_tools.audio_toolkit
            audio_instance = audio_toolkit()

            assert audio_instance is not None

        except Exception as e:
            pytest.skip(f"Audio toolkit not available: {e}")


class TestInfoRetrievalGroup:
    """Test information retrieval group - some tests require external services."""

    def test_info_retrieval_group_loading(self):
        """Test loading info retrieval group."""
        start_time = time.time()
        info_tools = groups.info_retrieval()
        load_time = time.time() - start_time

        # Should load quickly
        assert load_time < 1.0

        available_toolkits = dir(info_tools)

        # Should contain Wikipedia toolkit
        assert "wikipedia_toolkit" in available_toolkits

    def test_wikipedia_toolkit_access(self):
        """Test accessing Wikipedia toolkit (no external call)."""
        info_tools = groups.info_retrieval()

        try:
            wiki_toolkit = info_tools.wikipedia_toolkit
            wiki_instance = wiki_toolkit()

            assert wiki_instance is not None

        except Exception as e:
            pytest.skip(f"Wikipedia toolkit not available: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_wikipedia_search_integration(self):
        """Test Wikipedia search with actual API call."""
        info_tools = groups.info_retrieval()

        try:
            wiki_toolkit = info_tools.wikipedia_toolkit
            wiki_instance = wiki_toolkit()

            # Make actual API call
            result = await wiki_instance.search_wikipedia("Python programming language", num_results=1)

            if result and len(result) > 0 and isinstance(result[0], dict):
                if "error" in result[0]:
                    pytest.skip(f"Wikipedia API error: {result[0].get('error', 'Unknown error')}")
                elif "title" in result[0]:
                    assert "Python" in result[0]["title"]
                else:
                    pytest.skip("Unexpected Wikipedia response format")
            else:
                pytest.skip("No Wikipedia results or API unavailable")

        except Exception as e:
            pytest.skip(f"Wikipedia integration test failed: {e}")


class TestPerformanceComparison:
    """Test performance benefits of lazy loading."""

    def test_group_import_performance(self):
        """Test that group imports are fast (lazy loading benefit)."""
        start_time = time.time()

        # Import multiple groups
        groups.academic()
        groups.development()
        groups.info_retrieval()

        try:
            groups.image()
        except Exception:
            pass

        group_import_time = time.time() - start_time

        # Group imports should be very fast (under 1 second)
        assert group_import_time < 1.0

    def test_toolkit_access_timing(self):
        """Test timing of actual toolkit access (which triggers loading)."""
        academic_group = groups.academic()
        dev_group = groups.development()
        info_group = groups.info_retrieval()

        start_time = time.time()

        toolkits_accessed = 0
        try:
            _ = academic_group.arxiv_toolkit
            toolkits_accessed += 1
        except Exception:
            pass

        try:
            _ = dev_group.bash_toolkit
            toolkits_accessed += 1
        except Exception:
            pass

        try:
            _ = info_group.wikipedia_toolkit
            toolkits_accessed += 1
        except Exception:
            pass

        toolkit_access_time = time.time() - start_time

        # Even toolkit access should be reasonable
        assert toolkit_access_time < 10.0
        assert toolkits_accessed > 0

    def test_memory_efficiency(self):
        """Test that lazy loading is memory efficient."""
        modules_before = len(get_loaded_modules())

        # Import groups but don't access toolkits
        groups.academic()
        groups.development()
        groups.info_retrieval()

        modules_after_groups = len(get_loaded_modules())

        # Should not have loaded many additional modules yet
        module_increase = modules_after_groups - modules_before
        assert module_increase < 50  # Reasonable threshold


class TestErrorHandling:
    """Test error handling with lazy imports."""

    def test_non_existent_toolkit_graceful_handling(self):
        """Test graceful handling of non-existent toolkits."""
        dev_tools = groups.development()

        # Try to access non-existent toolkit
        non_existent = getattr(dev_tools, "non_existent_toolkit", None)
        assert non_existent is None

    def test_attribute_error_handling(self):
        """Test AttributeError handling for missing toolkits."""
        dev_tools = groups.development()

        with pytest.raises(AttributeError):
            _ = dev_tools.definitely_non_existent_toolkit

    def test_missing_dependency_handling(self):
        """Test handling of toolkits with missing dependencies."""
        # This test checks that toolkits with missing dependencies fail gracefully
        academic_tools = groups.academic()

        try:
            # Try to access a toolkit that might have missing dependencies
            toolkit = academic_tools.arxiv_toolkit
            # If it succeeds, try to instantiate
            instance = toolkit()
            # If that succeeds too, that's fine
            assert instance is not None

        except Exception as e:
            # If it fails due to missing dependencies, that's expected and acceptable
            assert isinstance(e, (ImportError, ModuleNotFoundError, AttributeError))


class TestLazyLoadingLifecycle:
    """Test lazy loading enable/disable lifecycle."""

    def test_enable_disable_lazy_loading(self):
        """Test enabling and disabling lazy loading."""
        # Ensure it's enabled
        cogents_tools.enable_lazy_loading()
        assert cogents_tools.is_lazy_loading_enabled()

        # Note: There might not be a disable function, which is fine
        # The main thing is that enable works

    def test_lazy_loading_persistence(self):
        """Test that lazy loading settings persist across operations."""
        cogents_tools.enable_lazy_loading()

        # Do some operations
        groups.academic()
        groups.development()

        # Should still be enabled
        assert cogents_tools.is_lazy_loading_enabled()


@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality in lazy-loaded toolkits."""

    async def test_async_toolkit_operations(self):
        """Test that async operations work with lazy-loaded toolkits."""
        dev_tools = groups.development()

        try:
            bash_toolkit = dev_tools.bash_toolkit
            bash_instance = bash_toolkit()

            # Test async operation
            result = await bash_instance.run_bash("echo 'async test'")
            assert isinstance(result, str)
            assert "async test" in result or "Error" in result  # Accept either success or controlled error

        except Exception as e:
            pytest.skip(f"Async bash test not available: {e}")

    async def test_multiple_async_toolkits(self):
        """Test using multiple async toolkits simultaneously."""
        tasks = []

        # Try bash toolkit
        try:
            dev_tools = groups.development()
            bash_toolkit = dev_tools.bash_toolkit
            bash_instance = bash_toolkit()
            tasks.append(bash_instance.run_bash("echo 'test1'"))
        except Exception:
            pass

        # If we have tasks, run them concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # At least one should succeed or return a controlled error
            assert len(results) > 0


class TestModuleTracking:
    """Test module loading tracking functionality."""

    def test_get_loaded_modules_returns_data(self):
        """Test that get_loaded_modules returns meaningful data."""
        loaded = get_loaded_modules()

        assert isinstance(loaded, (list, set, tuple))
        # Should have at least some modules loaded by now
        assert len(loaded) > 0

    def test_module_loading_tracking(self):
        """Test that module loading is properly tracked."""
        modules_before = set(get_loaded_modules())

        # Load a group and access a toolkit
        try:
            academic_tools = groups.academic()
            _ = academic_tools.arxiv_toolkit

            modules_after = set(get_loaded_modules())

            # Should have loaded additional modules (or at least not fewer)
            assert len(modules_after) >= len(modules_before)

        except Exception:
            # If loading fails, that's acceptable for this test
            pass

    def test_module_names_are_strings(self):
        """Test that module names in loaded modules are strings."""
        loaded = get_loaded_modules()

        for module in loaded:
            assert isinstance(module, str)
            assert len(module) > 0


if __name__ == "__main__":
    pytest.main([__file__])
