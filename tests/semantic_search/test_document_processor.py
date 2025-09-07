"""
Unit tests for document processor module.

Tests cover DocumentProcessor, ChunkingConfig, ProcessedDocument classes
and their various methods and edge cases.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from cogents_tools.integrations.semantic_search.docproc import ChunkingConfig, DocumentProcessor, ProcessedDocument
from cogents_tools.integrations.semantic_search.models import DocumentChunk


@pytest.fixture
def sample_chunking_config():
    """Provide a sample chunking configuration for testing."""
    return ChunkingConfig(chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ". ", "。", "！", "？"])


@pytest.fixture
def sample_document_processor(sample_chunking_config):
    """Provide a sample document processor for testing."""
    return DocumentProcessor(sample_chunking_config)


@pytest.fixture
def sample_document_chunk():
    """Provide a sample document chunk for testing."""
    return DocumentChunk(
        chunk_id="test-chunk-1",
        content="This is a sample document chunk for testing purposes.",
        metadata={"test": "value", "chunk_length": 50},
        source_url="http://example.com/test",
        source_title="Test Document",
        chunk_index=0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_search_result():
    """Provide a sample search result for testing."""
    result = Mock()
    result.url = "http://example.com/article"
    result.title = "Sample Article Title"
    result.content = "This is a sample article content for testing. " * 10
    result.raw_content = None
    result.score = 0.85
    return result


@pytest.fixture
def sample_search_response(sample_search_result):
    """Provide a sample search response for testing."""
    response = Mock()
    response.results = [sample_search_result]
    return response


@pytest.fixture
def sample_long_text():
    """Provide sample long text for chunking tests."""
    return (
        """
    This is a sample long text that will be used for testing document chunking.
    It contains multiple sentences and paragraphs to ensure proper splitting.
    
    The text should be long enough to create multiple chunks when processed.
    Each sentence should be properly handled by the chunking algorithm.
    
    We also include some Chinese text to test multilingual support: 这是一个测试文本。
    它包含中文内容来测试多语言支持。每个句子都应该被正确处理。
    
    The chunking should work with both English and Chinese punctuation marks.
    """
        * 5
    )


class TestChunkingConfig:
    """Test cases for ChunkingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.length_function == "len"
        assert config.is_separator_regex is False
        assert len(config.separators) > 0
        assert "\n\n" in config.separators
        assert "。" in config.separators  # Chinese punctuation

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n", ". "],
            length_function="len",
            is_separator_regex=True,
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.separators == ["\n", ". "]
        assert config.is_separator_regex is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Test chunk_size validation
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=50)  # Too small

        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=6000)  # Too large

        # Test chunk_overlap validation
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_overlap=-1)  # Negative overlap


class TestProcessedDocument:
    """Test cases for ProcessedDocument class."""

    def test_processed_document_creation(self):
        """Test creating a ProcessedDocument with basic data."""
        chunks = [
            DocumentChunk(
                chunk_id="test-1",
                content="Test content 1",
                metadata={},
                source_url="http://example.com",
                source_title="Test Title",
                chunk_index=0,
                timestamp=datetime.now(),
            )
        ]

        doc = ProcessedDocument(
            source_url="http://example.com",
            source_title="Test Title",
            total_chunks=1,
            chunks=chunks,
            metadata={"test": "value"},
        )

        assert doc.source_url == "http://example.com"
        assert doc.source_title == "Test Title"
        assert doc.total_chunks == 1
        assert len(doc.chunks) == 1
        assert doc.metadata["test"] == "value"
        assert isinstance(doc.processing_timestamp, datetime)

    def test_processed_document_defaults(self):
        """Test ProcessedDocument with default values."""
        doc = ProcessedDocument(
            source_url="http://example.com",
            source_title="Test Title",
            total_chunks=0,
            chunks=[],
        )

        assert doc.metadata == {}
        assert isinstance(doc.processing_timestamp, datetime)


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        self.processor = DocumentProcessor(self.config)

    def test_initialization_with_config(self):
        """Test DocumentProcessor initialization with custom config."""
        processor = DocumentProcessor(self.config)
        assert processor.config == self.config
        assert processor.config.chunk_size == 100

    def test_initialization_without_config(self):
        """Test DocumentProcessor initialization with default config."""
        processor = DocumentProcessor()
        assert processor.config.chunk_size == 1000
        assert processor.config.chunk_overlap == 200

    def test_create_text_splitter(self):
        """Test text splitter creation."""
        splitter = self.processor._create_text_splitter()
        # Check that the splitter was created successfully
        assert splitter is not None
        assert hasattr(splitter, "split_text")
        # The actual attributes are not directly accessible, but we can test the functionality

    def test_select_content_prefers_raw_content(self):
        """Test content selection prefers raw_content when available."""
        search_result = Mock()
        search_result.raw_content = "Raw content with substantial length that exceeds the minimum threshold for selection. This additional text ensures we have more than 100 characters to meet the requirement for raw_content selection."
        search_result.content = "Regular content"
        search_result.title = "Title"

        content = self.processor._select_content(search_result)
        assert (
            content
            == "Raw content with substantial length that exceeds the minimum threshold for selection. This additional text ensures we have more than 100 characters to meet the requirement for raw_content selection."
        )

    def test_select_content_falls_back_to_content(self):
        """Test content selection falls back to content when raw_content is short."""
        search_result = Mock()
        search_result.raw_content = "Short"
        search_result.content = "Regular content"
        search_result.title = "Title"

        content = self.processor._select_content(search_result)
        assert content == "Regular content"

    def test_select_content_falls_back_to_title(self):
        """Test content selection falls back to title when other content is unavailable."""
        search_result = Mock()
        search_result.raw_content = None
        search_result.content = None
        search_result.title = "Title"

        content = self.processor._select_content(search_result)
        assert content == "Title"

    def test_select_content_returns_empty_string(self):
        """Test content selection returns empty string when no content available."""
        search_result = Mock()
        search_result.raw_content = None
        search_result.content = None
        search_result.title = None

        content = self.processor._select_content(search_result)
        assert content == ""

    def test_create_chunks_from_content(self):
        """Test chunk creation from content."""
        content = "This is a test content. It should be split into chunks. " * 10
        metadata = {"test": "value"}

        chunks = self.processor._create_chunks_from_content(
            content=content,
            source_url="http://example.com",
            source_title="Test Title",
            metadata=metadata,
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.source_url == "http://example.com" for chunk in chunks)
        assert all(chunk.source_title == "Test Title" for chunk in chunks)
        assert all(chunk.metadata["test"] == "value" for chunk in chunks)
        assert all(chunk.metadata["chunk_length"] > 0 for chunk in chunks)

    def test_create_chunks_from_empty_content(self):
        """Test chunk creation from empty content."""
        chunks = self.processor._create_chunks_from_content(
            content="", source_url="http://example.com", source_title="Test Title"
        )

        assert len(chunks) == 0

    def test_process_search_result_success(self):
        """Test successful processing of a search result."""
        search_result = Mock()
        search_result.url = "http://example.com"
        search_result.title = "Test Title"
        search_result.content = "This is test content. " * 20
        search_result.raw_content = None
        search_result.score = 0.8

        processed_doc = self.processor.process_search_result(search_result)

        assert processed_doc is not None
        assert processed_doc.source_url == "http://example.com"
        assert processed_doc.source_title == "Test Title"
        assert processed_doc.total_chunks > 0
        assert len(processed_doc.chunks) > 0
        assert processed_doc.metadata["original_score"] == 0.8

    def test_process_search_result_insufficient_content(self):
        """Test processing search result with insufficient content."""
        search_result = Mock()
        search_result.url = "http://example.com"
        search_result.title = "Test Title"
        search_result.content = "Short"
        search_result.raw_content = None
        search_result.score = 0.8

        processed_doc = self.processor.process_search_result(search_result)

        assert processed_doc is None

    def test_process_search_result_exception(self):
        """Test processing search result that raises an exception."""
        search_result = Mock()
        search_result.url = "http://example.com"
        search_result.title = "Test Title"
        search_result.content = "This is test content. " * 20
        search_result.raw_content = None
        search_result.score = 0.8

        # Mock _create_chunks_from_content to raise an exception
        with patch.object(
            self.processor,
            "_create_chunks_from_content",
            side_effect=Exception("Test error"),
        ):
            processed_doc = self.processor.process_search_result(search_result)
            assert processed_doc is None

    def test_process_search_response(self):
        """Test processing a search response."""
        # Create mock search results
        result1 = Mock()
        result1.url = "http://example1.com"
        result1.title = "Title 1"
        result1.content = "This is test content for result 1. " * 20
        result1.raw_content = None
        result1.score = 0.8

        result2 = Mock()
        result2.url = "http://example2.com"
        result2.title = "Title 2"
        result2.content = "This is test content for result 2. " * 20
        result2.raw_content = None
        result2.score = 0.9

        search_response = Mock()
        search_response.results = [result1, result2]

        processed_docs = self.processor.process_search_response(search_response)

        assert len(processed_docs) == 2
        assert all(isinstance(doc, ProcessedDocument) for doc in processed_docs)

    def test_process_search_response_with_failures(self):
        """Test processing search response with some failures."""
        # Create mock search results
        result1 = Mock()
        result1.url = "http://example1.com"
        result1.title = "Title 1"
        result1.content = "This is test content for result 1. " * 20
        result1.raw_content = None
        result1.score = 0.8

        result2 = Mock()
        result2.url = "http://example2.com"
        result2.title = "Title 2"
        result2.content = "Short"  # This will be skipped
        result2.raw_content = None
        result2.score = 0.9

        search_response = Mock()
        search_response.results = [result1, result2]

        processed_docs = self.processor.process_search_response(search_response)

        assert len(processed_docs) == 1  # Only one successful result

    def test_process_search_response_with_exception(self):
        """Test processing search response with exception handling."""
        # Create a mock search result that will raise an exception
        result = Mock()
        result.url = "http://example.com"
        result.title = "Title"
        result.content = "This is test content. " * 20
        result.raw_content = None
        result.score = 0.8

        # Mock process_search_result to raise an exception
        with patch.object(
            self.processor,
            "process_search_result",
            side_effect=Exception("Test exception"),
        ):
            search_response = Mock()
            search_response.results = [result]

            processed_docs = self.processor.process_search_response(search_response)

            assert len(processed_docs) == 0  # No successful results due to exception

    def test_process_raw_text(self):
        """Test processing raw text content."""
        text = "This is raw text content. It should be processed into chunks. " * 10
        metadata = {"source": "manual"}

        processed_doc = self.processor.process_raw_text(
            text=text,
            source_url="http://example.com",
            source_title="Raw Text Title",
            metadata=metadata,
        )

        assert processed_doc.source_url == "http://example.com"
        assert processed_doc.source_title == "Raw Text Title"
        assert processed_doc.total_chunks > 0
        assert len(processed_doc.chunks) > 0
        assert processed_doc.metadata["source"] == "manual"

    def test_merge_chunks(self):
        """Test merging small chunks together."""
        # Create small chunks
        chunks = []
        for i in range(5):
            chunk = DocumentChunk(
                chunk_id=f"chunk-{i}",
                content=f"Small chunk {i}",
                metadata={"chunk_length": 12},
                source_url="http://example.com",
                source_title="Test Title",
                chunk_index=i,
                timestamp=datetime.now(),
            )
            chunks.append(chunk)

        merged_chunks = self.processor.merge_chunks(chunks, max_size=100)

        assert len(merged_chunks) < len(chunks)  # Should merge some chunks
        assert all(isinstance(chunk, DocumentChunk) for chunk in merged_chunks)

    def test_merge_chunks_empty_list(self):
        """Test merging empty list of chunks."""
        merged_chunks = self.processor.merge_chunks([])
        assert merged_chunks == []

    def test_merge_chunks_single_chunk(self):
        """Test merging single chunk."""
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            content="Single chunk",
            metadata={"chunk_length": 12},
            source_url="http://example.com",
            source_title="Test Title",
            chunk_index=0,
            timestamp=datetime.now(),
        )

        merged_chunks = self.processor.merge_chunks([chunk])
        assert len(merged_chunks) == 1
        assert merged_chunks[0] == chunk

    def test_get_stats(self):
        """Test getting processing statistics."""
        stats = self.processor.get_stats()

        assert "chunk_size" in stats
        assert "chunk_overlap" in stats
        assert "separators" in stats
        assert "text_splitter_type" in stats
        assert stats["chunk_size"] == 100
        assert stats["chunk_overlap"] == 20
        assert "RecursiveCharacterTextSplitter" in stats["text_splitter_type"]


class TestDocumentProcessorEdgeCases:
    """Test edge cases and error conditions."""

    def test_process_search_result_none_content(self):
        """Test processing search result with None content."""
        processor = DocumentProcessor()

        search_result = Mock()
        search_result.url = "http://example.com"
        search_result.title = "Test Title"
        search_result.content = None
        search_result.raw_content = None
        search_result.score = 0.8

        processed_doc = processor.process_search_result(search_result)
        assert processed_doc is None

    def test_process_search_result_empty_strings(self):
        """Test processing search result with empty string content."""
        processor = DocumentProcessor()

        search_result = Mock()
        search_result.url = "http://example.com"
        search_result.title = "Test Title"
        search_result.content = ""
        search_result.raw_content = ""
        search_result.score = 0.8

        processed_doc = processor.process_search_result(search_result)
        assert processed_doc is None

    def test_process_search_result_whitespace_only(self):
        """Test processing search result with whitespace-only content."""
        processor = DocumentProcessor()

        search_result = Mock()
        search_result.url = "http://example.com"
        search_result.title = "Test Title"
        search_result.content = "   \n\t   "
        search_result.raw_content = None
        search_result.score = 0.8

        processed_doc = processor.process_search_result(search_result)
        assert processed_doc is None

    def test_chunking_with_chinese_text(self):
        """Test chunking with Chinese text content."""
        processor = DocumentProcessor(ChunkingConfig(chunk_size=100, chunk_overlap=10))

        chinese_text = "这是一个中文测试文本。它包含多个句子。每个句子都应该被正确处理。"

        chunks = processor._create_chunks_from_content(
            content=chinese_text,
            source_url="http://example.com",
            source_title="Chinese Test",
            metadata={},
        )

        assert len(chunks) > 0
        assert all(len(chunk.content) > 0 for chunk in chunks)

    def test_chunking_with_mixed_language_text(self):
        """Test chunking with mixed English and Chinese text."""
        processor = DocumentProcessor(ChunkingConfig(chunk_size=100, chunk_overlap=20))

        mixed_text = "This is English text. 这是中文文本。Mixed content should work. 混合内容应该有效。"

        chunks = processor._create_chunks_from_content(
            content=mixed_text,
            source_url="http://example.com",
            source_title="Mixed Language Test",
            metadata={},
        )

        assert len(chunks) > 0
        assert all(len(chunk.content) > 0 for chunk in chunks)


class TestDocumentProcessorIntegration:
    """Integration tests for DocumentProcessor."""

    def test_full_processing_pipeline(self):
        """Test the complete processing pipeline from search response to chunks."""
        processor = DocumentProcessor(ChunkingConfig(chunk_size=200, chunk_overlap=50))

        # Create a realistic search response
        result = Mock()
        result.url = "http://example.com/article"
        result.title = "Sample Article Title"
        result.content = "This is a sample article content. " * 30
        result.raw_content = None
        result.score = 0.95

        search_response = Mock()
        search_response.results = [result]

        # Process the search response
        processed_docs = processor.process_search_response(search_response)

        assert len(processed_docs) == 1
        doc = processed_docs[0]

        # Verify the processed document
        assert doc.source_url == "http://example.com/article"
        assert doc.source_title == "Sample Article Title"
        assert doc.total_chunks > 0
        assert len(doc.chunks) > 0

        # Verify chunk properties
        for chunk in doc.chunks:
            assert chunk.chunk_id is not None
            assert len(chunk.content) > 0
            assert chunk.source_url == "http://example.com/article"
            assert chunk.source_title == "Sample Article Title"
            assert chunk.chunk_index >= 0
            assert isinstance(chunk.timestamp, datetime)
            assert "chunk_length" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_merge_chunks_integration(self):
        """Test integration of chunk merging with processing."""
        processor = DocumentProcessor(ChunkingConfig(chunk_size=100, chunk_overlap=10))

        # Process text that will create small chunks
        text = "Short sentence. Another short one. Third sentence. Fourth one. Fifth sentence."

        processed_doc = processor.process_raw_text(
            text=text, source_url="http://example.com", source_title="Test Article"
        )

        # Merge the chunks
        merged_chunks = processor.merge_chunks(processed_doc.chunks, max_size=100)

        # Verify merging worked
        assert len(merged_chunks) <= len(processed_doc.chunks)
        assert all(isinstance(chunk, DocumentChunk) for chunk in merged_chunks)

        # Verify content is preserved
        original_content = " ".join(chunk.content for chunk in processed_doc.chunks)
        merged_content = " ".join(chunk.content for chunk in merged_chunks)
        assert original_content.replace(" ", "") == merged_content.replace(" ", "")
