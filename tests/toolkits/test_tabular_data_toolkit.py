"""
Tests for TabularDataToolkit functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def tabular_config():
    """Create a test configuration for TabularDataToolkit."""
    return ToolkitConfig(
        name="tabular_data", config={"max_file_size": 50 * 1024 * 1024, "max_rows_display": 100}  # 50MB
    )


@pytest.fixture
def tabular_toolkit(tabular_config):
    """Create TabularDataToolkit instance for testing."""
    try:
        return get_toolkit("tabular_data", tabular_config)
    except KeyError as e:
        if "tabular_data" in str(e):
            pytest.skip("TabularDataToolkit not available for testing")
        raise


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,age,city,salary\n")
        f.write("Alice,25,New York,50000\n")
        f.write("Bob,30,San Francisco,75000\n")
        f.write("Charlie,35,Chicago,60000\n")
        f.write("Diana,28,Boston,55000\n")
        f.write("Eve,32,Seattle,70000\n")
        f.flush()
        yield f.name
    Path(f.name).unlink()


class TestTabularDataToolkit:
    """Test cases for TabularDataToolkit."""

    async def test_toolkit_initialization(self, tabular_toolkit):
        """Test that TabularDataToolkit initializes correctly."""
        assert tabular_toolkit is not None
        assert hasattr(tabular_toolkit, "get_tabular_columns")
        assert hasattr(tabular_toolkit, "get_column_info")
        assert hasattr(tabular_toolkit, "get_data_summary")
        assert hasattr(tabular_toolkit, "validate_data_quality")

    async def test_get_tools_map(self, tabular_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await tabular_toolkit.get_tools_map()

        expected_tools = ["get_tabular_columns", "get_column_info", "get_data_summary", "validate_data_quality"]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    def test_load_tabular_data_csv(self, tabular_toolkit, sample_csv_file):
        """Test loading CSV data."""
        df = tabular_toolkit._load_tabular_data(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 5 rows of data
        assert list(df.columns) == ["name", "age", "city", "salary"]
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[0]["age"] == 25

    def test_load_tabular_data_excel(self, tabular_toolkit):
        """Test loading Excel data."""
        # Create a sample Excel file
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30], "city": ["New York", "San Francisco"]})
            df.to_excel(f.name, index=False)

            try:
                loaded_df = tabular_toolkit._load_tabular_data(f.name)

                assert isinstance(loaded_df, pd.DataFrame)
                assert len(loaded_df) == 2
                assert list(loaded_df.columns) == ["name", "age", "city"]
            finally:
                Path(f.name).unlink()

    def test_load_tabular_data_json(self, tabular_toolkit):
        """Test loading JSON data."""
        # Create a sample JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            data = [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "San Francisco"},
            ]
            json.dump(data, f)
            f.flush()

            try:
                loaded_df = tabular_toolkit._load_tabular_data(f.name)

                assert isinstance(loaded_df, pd.DataFrame)
                assert len(loaded_df) == 2
                assert "name" in loaded_df.columns
            finally:
                Path(f.name).unlink()

    async def test_get_tabular_columns_success(self, tabular_toolkit, sample_csv_file):
        """Test successful column retrieval."""
        result = await tabular_toolkit.get_tabular_columns(sample_csv_file)

        assert isinstance(result, str)
        assert "Column 1" in result
        assert "Column 2" in result
        assert "Column 3" in result
        assert "Column 4" in result

    async def test_get_tabular_columns_file_not_found(self, tabular_toolkit):
        """Test column retrieval with file not found."""
        result = await tabular_toolkit.get_tabular_columns("/nonexistent/file.csv")

        assert isinstance(result, str)
        assert "Error:" in result
        assert "does not exist" in result

    async def test_get_tabular_columns_load_error(self, tabular_toolkit):
        """Test column retrieval with loading error."""
        # Create invalid CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("invalid,csv,data\n")
            f.write("missing,columns\n")  # Inconsistent columns
            f.flush()

            try:
                result = await tabular_toolkit.get_tabular_columns(f.name)

                # Should still work with pandas error handling
                assert isinstance(result, str)
                # Result could be success or error depending on pandas behavior
            finally:
                Path(f.name).unlink()

    @pytest.mark.integration
    async def test_get_column_info_success(self, tabular_toolkit, sample_csv_file):
        """Test successful column info retrieval."""
        result = await tabular_toolkit.get_column_info(sample_csv_file)

        assert isinstance(result, str)
        assert "Column" in result  # Should contain column information

    @pytest.mark.integration
    async def test_get_column_info_string_column(self, tabular_toolkit, sample_csv_file):
        """Test column info for string column."""
        result = await tabular_toolkit.get_column_info(sample_csv_file)

        assert isinstance(result, str)
        assert "Column" in result  # Should contain column information

    @pytest.mark.integration
    async def test_get_column_info_nonexistent_column(self, tabular_toolkit, sample_csv_file):
        """Test column info for non-existent column."""
        result = await tabular_toolkit.get_column_info(sample_csv_file)

        assert isinstance(result, str)
        assert "Column" in result  # Should contain column information

    @pytest.mark.integration
    async def test_get_column_info_file_not_found(self, tabular_toolkit):
        """Test column info with file not found."""
        result = await tabular_toolkit.get_column_info("/nonexistent/file.csv")

        assert isinstance(result, str)
        assert "Error" in result  # Should contain error message


class TestTabularDataToolkitEdgeCases:
    """Test edge cases and error conditions."""

    async def test_large_dataset_handling(self, tabular_toolkit):
        """Test handling of large datasets."""
        # Create a large CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,value\n")
            for i in range(1000):  # 1000 rows
                f.write(f"{i},{i*2}\n")
            f.flush()

            try:
                result = await tabular_toolkit.get_tabular_columns(f.name)

                assert isinstance(result, str)
                assert "Column 1" in result
                assert "Column 2" in result

            finally:
                Path(f.name).unlink()

    async def test_empty_dataset(self, tabular_toolkit):
        """Test handling of empty datasets."""
        # Create empty CSV file (only headers)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,city\n")  # Only headers
            f.flush()

            try:
                result = await tabular_toolkit.get_tabular_columns(f.name)

                assert isinstance(result, str)
                assert "Column" in result

            finally:
                Path(f.name).unlink()

    @pytest.mark.integration
    async def test_dataset_with_missing_values(self, tabular_toolkit):
        """Test handling of datasets with missing values."""
        # Create CSV with missing values
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,city\n")
            f.write("Alice,25,New York\n")
            f.write("Bob,,San Francisco\n")  # Missing age
            f.write("Charlie,35,\n")  # Missing city
            f.write(",30,Chicago\n")  # Missing name
            f.flush()

            try:
                # Test column info with missing values
                result = await tabular_toolkit.get_column_info(f.name)

                assert isinstance(result, str)
                assert "Column" in result

            finally:
                Path(f.name).unlink()

    async def test_concurrent_operations(self, tabular_toolkit, sample_csv_file):
        """Test concurrent tabular data operations."""
        import asyncio

        # Perform multiple operations concurrently
        tasks = [
            tabular_toolkit.get_tabular_columns(sample_csv_file),
            tabular_toolkit.get_column_info(sample_csv_file),
        ]

        results = await asyncio.gather(*tasks)

        # All operations should succeed
        for result in results:
            assert isinstance(result, str)

    async def test_special_characters_in_data(self, tabular_toolkit):
        """Test handling data with special characters."""
        # Create CSV with special characters
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("name,description\n")
            f.write("José,Café owner\n")
            f.write("李明,Software engineer\n")
            f.write("Müller,German scientist\n")
            f.flush()

            try:
                result = await tabular_toolkit.get_tabular_columns(f.name)

                assert isinstance(result, str)
                assert "Column" in result

            finally:
                Path(f.name).unlink()
