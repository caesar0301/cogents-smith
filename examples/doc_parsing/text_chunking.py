"""
Text Chunking Example

This example demonstrates how to split text into chunks using StandardChunker.
"""

import asyncio

from cogent_base.sensory.chunker import StandardChunker


async def main():
    # Initialize the chunker
    chunker = StandardChunker(chunk_size=1000, overlap=200)

    # Sample long text
    long_text = (
        """
    This is a very long document that needs to be split into smaller chunks
    for processing. The chunker will split this text into manageable pieces
    while maintaining some overlap between chunks to preserve context.

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
    veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
    commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
    velit esse cillum dolore eu fugiat nulla pariatur.

    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui
    officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde
    omnis iste natus error sit voluptatem accusantium doloremque laudantium.
    """
        * 10
    )  # Repeat to make it longer

    try:
        # Split the text into chunks
        chunks = await chunker.split_text(long_text)

        print(f"Original text length: {len(long_text)} characters")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Chunk size: {chunker.chunk_size}, Overlap: {chunker.overlap}")

        # Display first few chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} (length: {len(chunk)}):")
            print(f"{chunk[:200]}...")

    except Exception as e:
        print(f"Error chunking text: {e}")


if __name__ == "__main__":
    asyncio.run(main())
