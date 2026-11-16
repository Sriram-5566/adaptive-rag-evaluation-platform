
# ==========================================
# RAG EVALUATION LAB
# Four Baseline Chunking Methods
# ==========================================

import re


# METHOD 1: FIXED-SIZE CHUNKING

def fixed_size_chunking(text, chunk_size=100):
    """Split text into chunks of 100 words by default."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)

    return chunks


# METHOD 2: SENTENCE-BASED CHUNKING

def sentence_chunking(text, sentences_per_chunk=3):
    """Group consecutive sentences into chunks."""

    if sentences_per_chunk <= 0:
        raise ValueError(
            "Sentences per chunk must be positive"
        )

    # Simple sentence splitting for our first prototype
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    for start in range(
        0,
        len(sentences),
        sentences_per_chunk
    ):
        chunk = " ".join(
            sentences[start:start + sentences_per_chunk]
        )
        chunks.append(chunk)

    return chunks


# METHOD 3: PARAGRAPH-BASED CHUNKING


# METHOD 3: PARAGRAPH / SECTION-BASED CHUNKING


def paragraph_chunking(text):
    """
    Split the synthetic university-policy PDF
    using its known numbered section headings.
    """

    import re

    # These headings belong to our sample document
    headings = [
        "Course Withdrawal",
        "Parking Permits",
        "Library Access",
        "Assignment Submission",
        "Student Identification",
        "Academic Advising",
        "Campus Shuttle",
        "Research Lab Access"
    ]

    # Match numbered headings even when PDF extraction
    # puts them in the middle of a line.
    pattern = r"(?=\b\d+\.\s+(?:" + "|".join(
        re.escape(h) for h in headings
    ) + r")\b)"

    sections = re.split(pattern, text)

    chunks = [
        section.strip()
        for section in sections
        if section.strip()
    ]

    return chunks



# METHOD 4: OVERLAPPING CHUNKING

def overlapping_chunking(
    text,
    chunk_size=100,
    overlap=20
):
    """Split into word chunks with repeated context."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be between 0 "
            "and chunk_size - 1"
        )

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk = " ".join(
            words[start:start + chunk_size]
        )

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks
