import re


_HEADING_MARKER = re.compile(r"(?<!\S)#{1,6}[ \t]+\S")


def normalize_markdown_heading_lines(text: str) -> str:
    """Put Markdown headings on separate lines with blank lines around them."""
    parts: list[tuple[str, bool]] = []
    for line in text.splitlines():
        markers = list(_HEADING_MARKER.finditer(line))
        if not markers:
            parts.append((line.rstrip(), False))
            continue

        if line[:markers[0].start()].strip():
            parts.append((line[:markers[0].start()].rstrip(), False))

        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(line)
            heading = line[marker.start():end].strip()
            parts.append((heading, True))

    normalized: list[str] = []
    for index, (line, is_heading) in enumerate(parts):
        if is_heading and normalized and normalized[-1].strip():
            normalized.append("")
        normalized.append(line)
        if (
            is_heading
            and index + 1 < len(parts)
            and parts[index + 1][0].strip()
            and not parts[index + 1][1]
        ):
            normalized.append("")

    compacted: list[str] = []
    for line in normalized:
        if line.strip() or (compacted and compacted[-1].strip()):
            compacted.append(line)
    return "\n".join(compacted).strip()
