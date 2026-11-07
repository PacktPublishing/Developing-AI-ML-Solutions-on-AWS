# /// script
# dependencies = []
# ///
"""Split text into pieces an embedding model will accept.

Every embedding model has a window, and the local one is the smallest in this
chapter: mxbai-embed-large stops at 512 tokens where Titan Text Embeddings v2
takes 8,192. Text that fits the cloud and not the laptop is a chapter that only
runs in one place, so the split is sized for the smaller window and used by both.

Paragraph breaks come first, because a paragraph is a better unit to retrieve
than a fixed window. A page of a bank statement has no paragraph breaks at all,
only rows, so anything still over the cap is cut on line boundaries, and a single
line longer than the cap is cut where it has to be.
"""

# Characters, not tokens: a cheap proxy that needs no tokenizer, set low enough
# that about 800 characters of English stays inside a 512-token window.
CHUNK_SIZE = 800


def _hard_split(text: str, size: int) -> list[str]:
    """Cut one oversized piece on line boundaries, then anywhere it must."""
    pieces: list[str] = []
    current = ""
    for line in text.splitlines():
        while len(line) > size:
            # a single line longer than the cap: no boundary to prefer
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
        # + 2 for the blank line that rejoins them, as the memo corpus was
        # chunked before this was shared, so its chunks do not move
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    # paragraphs alone can still leave a piece over the window
    capped: list[str] = []
    for chunk in chunks:
        capped.extend([chunk] if len(chunk) <= size else _hard_split(chunk, size))
    return capped
