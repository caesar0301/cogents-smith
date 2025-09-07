"""
Tabular data toolkit for analyzing and processing structured data files.

Provides tools for reading, analyzing, and extracting insights from various
tabular data formats including CSV, Excel, JSON, and Parquet files.
"""

import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

# Template for column analysis
COLUMN_ANALYSIS_TEMPLATE = """You are a data analysis agent that extracts and summarizes data structure information from tabular data files (CSV, Excel, etc.).

<column_info>
{column_info}
</column_info>

You should extract the file structure (e.g. the delimiter), and provide detailed column information (e.g. column_name, type, column explanation and sample values) for each column.

<output_format>
### File Structure
- Delimiter: <the delimiter used in the file, e.g. ',', '\\t', ' '>

### Columns
| Column Name | Type | Explanation | Sample Value |
|-------------|------|-------------|--------------|
| name_of_column1 | type_of_column1 | explanation_of_column1, i.e. what the column represents | sample_value_of_column1 |
| name_of_column2 | type_of_column2 | explanation_of_column2, i.e. what the column represents | sample_value_of_column2 |
| ... | ... | ... | ... |
</output_format>"""


@register_toolkit("tabular_data")
class TabularDataToolkit(AsyncBaseToolkit):
    """
    Toolkit for tabular data analysis and processing.

    This toolkit provides capabilities for:
    - Reading various tabular data formats (CSV, Excel, JSON, Parquet, TSV)
    - Analyzing data structure and column information
    - Extracting metadata and statistics
    - LLM-powered data interpretation
    - Data quality assessment

    Features:
    - Multi-format support with automatic encoding detection
    - Intelligent column analysis and interpretation
    - Statistical summaries and data profiling
    - LLM-powered column explanation generation
    - Data validation and quality checks
    - Memory-efficient processing for large files

    Supported Formats:
    - **CSV**: Comma-separated values with encoding detection
    - **Excel**: XLSX and XLS files with multiple sheets
    - **JSON**: JSON files with tabular structure
    - **Parquet**: Columnar storage format
    - **TSV**: Tab-separated values
    - **Generic**: Auto-detection for other delimited formats

    Required dependency: pandas
    Install with: pip install pandas
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the tabular data toolkit.

        Args:
            config: Toolkit configuration

        Raises:
            ImportError: If pandas is not installed
        """
        super().__init__(config)

        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for TabularDataToolkit. " "Install with: pip install pandas")

        # Configuration
        self.max_file_size = self.config.config.get("max_file_size", 100 * 1024 * 1024)  # 100MB
        self.max_rows_preview = self.config.config.get("max_rows_preview", 1000)
        self.cache_dir = Path(self.config.config.get("cache_dir", "./tabular_cache"))

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Tabular data toolkit initialized")

    def _load_tabular_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load tabular data from a file with automatic format detection.

        Args:
            file_path: Path to the data file
            **kwargs: Additional parameters for pandas readers

        Returns:
            DataFrame containing the tabular data

        Raises:
            Exception: If the file cannot be loaded as tabular data
        """
        file_path = Path(file_path)

        # Check file size
        if file_path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large ({file_path.stat().st_size} bytes, max: {self.max_file_size})")

        file_ext = file_path.suffix.lower()

        try:
            if file_ext == ".csv":
                # Try different encodings for CSV files
                encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                        self.logger.info(f"Successfully loaded CSV with {encoding} encoding")
                        return df
                    except UnicodeDecodeError:
                        continue
                raise Exception("Could not read CSV file with any supported encoding")

            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, **kwargs)
                self.logger.info(f"Successfully loaded Excel file")
                return df

            elif file_ext == ".json":
                df = pd.read_json(file_path, **kwargs)
                self.logger.info(f"Successfully loaded JSON file")
                return df

            elif file_ext == ".parquet":
                df = pd.read_parquet(file_path, **kwargs)
                self.logger.info(f"Successfully loaded Parquet file")
                return df

            elif file_ext == ".tsv":
                # Tab-separated values
                encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, sep="\t", encoding=encoding, **kwargs)
                        self.logger.info(f"Successfully loaded TSV with {encoding} encoding")
                        return df
                    except UnicodeDecodeError:
                        continue
                raise Exception("Could not read TSV file with any supported encoding")

            else:
                # Try to read as CSV by default
                try:
                    df = pd.read_csv(file_path, **kwargs)
                    self.logger.info(f"Successfully loaded file as CSV")
                    return df
                except Exception as e:
                    raise Exception(f"Unsupported file format: {file_ext}") from e

        except Exception as e:
            self.logger.error(f"Failed to load tabular data: {e}")
            raise

    def _extract_column_info(self, df: pd.DataFrame, return_features: Optional[List[str]] = None) -> List[Dict]:
        """
        Extract detailed information about DataFrame columns.

        Args:
            df: DataFrame to analyze
            return_features: List of features to include in output

        Returns:
            List of dictionaries containing column information
        """
        column_info = []

        for col in df.columns:
            try:
                # Get data type
                dtype = str(df[col].dtype)

                # Get sample value (first non-null value)
                sample_value = None
                non_null_values = df[col].dropna()

                if len(non_null_values) > 0:
                    sample_value = non_null_values.iloc[0]

                    # Handle different data types for display
                    if pd.isna(sample_value):
                        sample_str = "NaN"
                    elif isinstance(sample_value, float):
                        if math.isnan(sample_value):
                            sample_str = "NaN"
                        else:
                            sample_str = str(sample_value)
                    else:
                        sample_str = str(sample_value)
                else:
                    sample_str = "No data"

                # Additional statistics
                null_count = df[col].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                unique_count = df[col].nunique()

                col_info = {
                    "column_name": str(col),
                    "type": dtype,
                    "sample": sample_str,
                    "null_count": int(null_count),
                    "null_percentage": round(null_percentage, 2),
                    "unique_count": int(unique_count),
                    "total_rows": len(df),
                }

                # Add numeric statistics for numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info.update(
                        {
                            "mean": df[col].mean() if not df[col].empty else None,
                            "std": df[col].std() if not df[col].empty else None,
                            "min": df[col].min() if not df[col].empty else None,
                            "max": df[col].max() if not df[col].empty else None,
                        }
                    )

                column_info.append(col_info)

            except Exception as e:
                self.logger.warning(f"Error processing column '{col}': {e}")
                column_info.append(
                    {"column_name": str(col), "type": "unknown", "sample": "Error reading sample", "error": str(e)}
                )

        return column_info

    def _format_column_info(self, column_info: List[Dict], return_features: Optional[List[str]] = None) -> str:
        """
        Format column information as a readable string.

        Args:
            column_info: List of column information dictionaries
            return_features: Features to include in output

        Returns:
            Formatted string representation
        """
        if not column_info:
            return "No columns found"

        if "error" in column_info[0]:
            return column_info[0]["error"]

        lines = []
        default_features = ["column_name", "type", "sample", "null_percentage", "unique_count"]
        features_to_show = return_features if return_features else default_features

        for i, col in enumerate(column_info):
            # Filter features to show
            filtered_info = {k: col[k] for k in features_to_show if k in col}

            lines.append(f"- Column {i + 1}: {json.dumps(filtered_info, ensure_ascii=False, default=str)}")

        return "\n".join(lines)

    async def get_tabular_columns(self, file_path: str, return_features: Optional[List[str]] = None) -> str:
        """
        Extract raw column metadata from tabular data files.

        This tool directly reads tabular data files and returns basic column
        information including names, data types, sample values, and statistics.
        It's useful for understanding the structure of data files before analysis.

        Args:
            file_path: Path to the tabular data file
            return_features: List of features to include (column_name, type, sample,
                           null_count, null_percentage, unique_count, mean, std, min, max)

        Returns:
            Formatted string with column information and statistics

        Supported formats: CSV, Excel (XLSX/XLS), JSON, Parquet, TSV

        Example:
            info = await get_tabular_columns("data.csv", ["column_name", "type", "sample"])
        """
        self.logger.info(f"Analyzing tabular columns for: {file_path}")

        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."

        try:
            # Load the data
            df = self._load_tabular_data(file_path)

            # Extract column information
            column_info = self._extract_column_info(df, return_features)

            # Format and return
            result = self._format_column_info(column_info, return_features)

            self.logger.info(f"Successfully analyzed {len(column_info)} columns")
            return result

        except Exception as e:
            error_msg = f"Error analyzing file '{file_path}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def get_column_info(self, file_path: str) -> str:
        """
        Get intelligent analysis and interpretation of column information.

        This tool builds on get_tabular_columns() to provide LLM-powered analysis
        of the data structure, including file format detection and column meaning
        interpretation. It's useful for understanding what the data represents.

        Args:
            file_path: Path to the tabular data file

        Returns:
            Detailed analysis with file structure and column explanations

        Example:
            analysis = await get_column_info("sales_data.csv")
            # Returns structured analysis with column explanations
        """
        self.logger.info(f"Getting intelligent column analysis for: {file_path}")

        try:
            # Get raw column information
            column_info_str = await self.get_tabular_columns(file_path)

            if column_info_str.startswith("Error:"):
                return column_info_str

            # Use LLM for intelligent analysis
            prompt = COLUMN_ANALYSIS_TEMPLATE.format(column_info=column_info_str)

            response = self.llm_client.completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analysis expert specializing in tabular data structure analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )

            return response.strip()

        except Exception as e:
            error_msg = f"Error during intelligent analysis: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def get_data_summary(self, file_path: str, max_rows: Optional[int] = None) -> str:
        """
        Get a comprehensive summary of the tabular data.

        Args:
            file_path: Path to the tabular data file
            max_rows: Maximum number of rows to analyze (default: 1000)

        Returns:
            Comprehensive data summary including statistics and insights
        """
        self.logger.info(f"Generating data summary for: {file_path}")

        max_rows = max_rows or self.max_rows_preview

        try:
            # Load data (with row limit for large files)
            df = self._load_tabular_data(file_path, nrows=max_rows)

            summary_lines = [
                f"Data Summary for: {file_path}",
                "=" * 50,
                "",
                f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
                f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                "",
            ]

            # Data types summary
            type_counts = df.dtypes.value_counts()
            summary_lines.append("Data Types:")
            for dtype, count in type_counts.items():
                summary_lines.append(f"  {dtype}: {count} columns")
            summary_lines.append("")

            # Missing data summary
            missing_data = df.isnull().sum()
            missing_cols = missing_data[missing_data > 0]
            if len(missing_cols) > 0:
                summary_lines.append("Missing Data:")
                for col, count in missing_cols.items():
                    percentage = (count / len(df)) * 100
                    summary_lines.append(f"  {col}: {count} ({percentage:.1f}%)")
            else:
                summary_lines.append("Missing Data: None")
            summary_lines.append("")

            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                summary_lines.append("Numeric Columns Summary:")
                desc = df[numeric_cols].describe()
                summary_lines.append(desc.to_string())
            summary_lines.append("")

            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                summary_lines.append("Categorical Columns (Top Values):")
                for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    top_values = df[col].value_counts().head(3)
                    summary_lines.append(f"  {col}:")
                    for value, count in top_values.items():
                        summary_lines.append(f"    {value}: {count}")
                summary_lines.append("")

            # Data quality indicators
            summary_lines.append("Data Quality Indicators:")
            duplicate_rows = df.duplicated().sum()
            summary_lines.append(f"  Duplicate rows: {duplicate_rows}")

            # Unique values per column
            unique_ratios = df.nunique() / len(df)
            high_cardinality = unique_ratios[unique_ratios > 0.9].index.tolist()
            if high_cardinality:
                summary_lines.append(f"  High cardinality columns: {', '.join(high_cardinality)}")

            return "\n".join(summary_lines)

        except Exception as e:
            error_msg = f"Error generating data summary: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def validate_data_quality(self, file_path: str) -> str:
        """
        Perform data quality validation and return a report.

        Args:
            file_path: Path to the tabular data file

        Returns:
            Data quality validation report
        """
        self.logger.info(f"Validating data quality for: {file_path}")

        try:
            df = self._load_tabular_data(file_path, nrows=self.max_rows_preview)

            issues = []

            # Check for missing data
            missing_data = df.isnull().sum()
            high_missing = missing_data[missing_data > len(df) * 0.5]
            if len(high_missing) > 0:
                issues.append(f"High missing data (>50%): {list(high_missing.index)}")

            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                issues.append(f"Duplicate rows found: {duplicates}")

            # Check for columns with single value
            single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
            if single_value_cols:
                issues.append(f"Columns with single/no values: {single_value_cols}")

            # Check for potential ID columns (high cardinality)
            potential_ids = [col for col in df.columns if df[col].nunique() == len(df)]
            if potential_ids:
                issues.append(f"Potential ID columns (unique values): {potential_ids}")

            # Check for mixed data types in object columns
            mixed_type_issues = []
            for col in df.select_dtypes(include=["object"]).columns:
                sample_types = df[col].dropna().apply(type).unique()
                if len(sample_types) > 1:
                    mixed_type_issues.append(col)
            if mixed_type_issues:
                issues.append(f"Columns with mixed data types: {mixed_type_issues}")

            # Generate report
            if issues:
                report = f"Data Quality Issues Found ({len(issues)}):\n\n"
                for i, issue in enumerate(issues, 1):
                    report += f"{i}. {issue}\n"
            else:
                report = "✅ No major data quality issues detected."

            return report

        except Exception as e:
            error_msg = f"Error during data quality validation: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "get_tabular_columns": self.get_tabular_columns,
            "get_column_info": self.get_column_info,
            "get_data_summary": self.get_data_summary,
            "validate_data_quality": self.validate_data_quality,
        }
