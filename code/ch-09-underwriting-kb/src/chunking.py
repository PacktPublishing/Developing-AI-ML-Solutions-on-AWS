# /// script
# dependencies = []
# ///
"""Split text into chunks with a maximum character count.

Prefer paragraph boundaries, then line boundaries. Split oversized lines at the
character limit. The character cap does not measure the embedding token count.
"""

# Maximum characters per chunk; token count depends on the embedding tokenizer.
CHUNK_SIZE = 800


def _hard_split(text: str, size: int) -> list[str]:
    """Split oversized text at line boundaries and the character limit."""
    pieces: list[str] = []
    current = ""
    for line in text.splitlines():
        while len(line) > size:
            # Split a line that exceeds the character limit.
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:size])
            line = line[size:]
        if not current:
            current = line
        elif len(current) + 1 + len(line) <= size:
            current = f"{current}\n{line}"
        else:
            pieces.append(current)
            current = line
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks of about `size` characters, none of them over it."""
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # Include the two newline characters between paragraphs.
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    # Split any paragraphs that still exceed the character limit.
    capped: list[str] = []
    for chunk in chunks:
        capped.extend([chunk] if len(chunk) <= size else _hard_split(chunk, size))
    return capped
