"""
Document Parsing Example

This example demonstrates how to parse documents using CogentParser.
"""

import asyncio

from cogent_base.sensory.parser import CogentParser


async def main():
    # Initialize the parser
    parser = CogentParser()

    # Example file content (in practice, you'd read from a file)
    sample_content = b"""
    This is a sample document content.
    It contains multiple paragraphs and various text elements.

    The parser will extract text and metadata from this content.
    """

    filename = "sample_document.txt"

    try:
        # Parse the file content
        metadata, elements = await parser.parse_file_to_text(sample_content, filename)

        print(f"Parsed document: {filename}")
        print(f"Metadata: {metadata}")
        print(f"Number of elements: {len(elements)}")

        # Display first few elements
        for i, element in enumerate(elements[:3]):
            print(f"Element {i}: {element}")

    except Exception as e:
        print(f"Error parsing document: {e}")


if __name__ == "__main__":
    asyncio.run(main())
