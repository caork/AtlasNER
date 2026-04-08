"""Entity and tag definitions."""

from __future__ import annotations

from atlas_ner.data.schemes import parse_tag


DEFAULT_ENTITY_DEFINITIONS: dict[str, dict[str, str]] = {
    "conll2003": {
        "PER": "A person, including an individual, public figure, artist, athlete, politician, or other named human being.",
        "ORG": "An organization such as a company, institution, government body, sports team, media outlet, or other named collective.",
        "LOC": "A named location such as a city, country, region, mountain, river, or other geographic place.",
        "MISC": "A named entity that does not fit person, organization, or location, such as an event, work of art, nationality, language, or product.",
    },
}


def get_entity_definition(
    dataset_name: str,
    entity_type: str,
    style: str = "long",
) -> str:
    del style
    dataset_key = dataset_name.lower()
    dataset_bank = DEFAULT_ENTITY_DEFINITIONS.get(dataset_key, {})
    if entity_type in dataset_bank:
        return dataset_bank[entity_type]
    return f"A named entity of type {entity_type}."


def build_tag_definition(
    tag_name: str,
    dataset_name: str,
    use_label_name_only: bool = False,
) -> str:
    if tag_name == "O":
        return "A token outside any named entity mention."
    prefix, entity_type = parse_tag(tag_name)
    if entity_type is None:
        return tag_name
    if use_label_name_only:
        prefix_map = {
            "B": "beginning",
            "I": "inside",
            "E": "ending",
            "S": "single-token",
        }
        prefix_text = prefix_map.get(prefix, prefix)
        return f"{prefix_text} tag for {entity_type}"

    entity_definition = get_entity_definition(dataset_name, entity_type)
    if prefix == "B":
        return f"The first token of a named entity. Entity type: {entity_definition}"
    if prefix == "I":
        return f"A middle token inside a multi-token named entity. Entity type: {entity_definition}"
    if prefix == "E":
        return f"The final token of a multi-token named entity. Entity type: {entity_definition}"
    if prefix == "S":
        return f"A complete single-token named entity. Entity type: {entity_definition}"
    return entity_definition


def build_tag_definitions(
    label_names: list[str],
    dataset_name: str,
    use_label_name_only: bool = False,
) -> dict[str, str]:
    return {
        label_name: build_tag_definition(
            tag_name=label_name,
            dataset_name=dataset_name,
            use_label_name_only=use_label_name_only,
        )
        for label_name in label_names
    }
