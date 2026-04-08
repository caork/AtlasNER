"""Tag-scheme utilities."""

from __future__ import annotations

from collections.abc import Iterable


def parse_tag(tag: str) -> tuple[str, str | None]:
    if tag == "O":
        return "O", None
    if "-" not in tag:
        raise ValueError(f"Malformed tag: {tag}")
    prefix, entity_type = tag.split("-", 1)
    return prefix, entity_type


def spans_to_tags(
    spans: list[tuple[str, int, int]],
    length: int,
    scheme: str,
) -> list[str]:
    scheme = scheme.upper()
    tags = ["O"] * length
    for entity_type, start, end in spans:
        if end - start <= 0:
            continue
        if scheme == "BIO":
            tags[start] = f"B-{entity_type}"
            for index in range(start + 1, end):
                tags[index] = f"I-{entity_type}"
            continue
        if scheme == "BIOES":
            span_len = end - start
            if span_len == 1:
                tags[start] = f"S-{entity_type}"
            else:
                tags[start] = f"B-{entity_type}"
                for index in range(start + 1, end - 1):
                    tags[index] = f"I-{entity_type}"
                tags[end - 1] = f"E-{entity_type}"
            continue
        raise ValueError(f"Unsupported scheme: {scheme}")
    return tags


def tags_to_spans(tags: Iterable[str]) -> list[tuple[str, int, int]]:
    sequence = list(tags)
    spans: list[tuple[str, int, int]] = []
    active_type: str | None = None
    active_start: int | None = None

    def close_entity(end_index: int) -> None:
        nonlocal active_type, active_start
        if active_type is not None and active_start is not None and active_start < end_index:
            spans.append((active_type, active_start, end_index))
        active_type = None
        active_start = None

    for index, tag in enumerate(sequence):
        prefix, entity_type = parse_tag(tag)
        if prefix == "O":
            close_entity(index)
            continue
        if prefix == "S":
            close_entity(index)
            spans.append((entity_type or "UNK", index, index + 1))
            continue
        if prefix == "B":
            close_entity(index)
            active_type = entity_type
            active_start = index
            continue
        if prefix == "I":
            if active_type != entity_type:
                close_entity(index)
                active_type = entity_type
                active_start = index
            continue
        if prefix == "E":
            if active_type == entity_type and active_start is not None:
                spans.append((entity_type or "UNK", active_start, index + 1))
                active_type = None
                active_start = None
            else:
                close_entity(index)
                spans.append((entity_type or "UNK", index, index + 1))
            continue
        raise ValueError(f"Unsupported prefix: {prefix}")

    close_entity(len(sequence))
    return spans


def convert_tag_scheme(tags: list[str], target_scheme: str) -> list[str]:
    target_scheme = target_scheme.upper()
    if target_scheme not in {"BIO", "BIOES"}:
        raise ValueError(f"Unsupported target scheme: {target_scheme}")
    spans = tags_to_spans(tags)
    return spans_to_tags(spans, len(tags), target_scheme)


def canonical_entity_types(tags: Iterable[str]) -> list[str]:
    entity_types = sorted({entity for _, entity in map(parse_tag, tags) if entity is not None})
    return entity_types
