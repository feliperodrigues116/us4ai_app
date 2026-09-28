"""Version-controlled frontend vocabulary for the US4AI study only."""

AI_TASK_CATALOG = {
    "Multimodal": (
        "Audio-Text-to-Text", "Image-Text-to-Text", "Image-Text-to-Image",
        "Image-Text-to-Video", "Visual Question Answering", "Document Question Answering",
        "Video-Text-to-Text", "Visual Document Retrieval", "Any-to-Any",
    ),
    "Computer Vision": (
        "Depth Estimation", "Image Classification", "Object Detection", "Image Segmentation",
        "Text-to-Image", "Image-to-Text", "Image-to-Image", "Image-to-Video",
        "Unconditional Image Generation", "Video Classification", "Text-to-Video",
        "Zero-Shot Image Classification", "Mask Generation", "Zero-Shot Object Detection",
        "Text-to-3D", "Image-to-3D", "Image Feature Extraction", "Keypoint Detection", "Video-to-Video",
    ),
    "Natural Language Processing": (
        "Text Classification", "Token Classification", "Table Question Answering",
        "Question Answering", "Zero-Shot Classification", "Translation", "Summarization",
        "Feature Extraction", "Text Generation", "Fill-Mask", "Sentence Similarity", "Text Ranking",
    ),
    "Audio": (
        "Text-to-Speech", "Text-to-Audio", "Automatic Speech Recognition", "Audio-to-Audio",
        "Audio Classification", "Voice Activity Detection", "Tabular",
    ),
    "Tabular Classification": ("Tabular Regression", "Time Series Forecasting"),
    "Reinforcement Learning": ("Reinforcement Learning", "Robotics"),
    "Other": ("Graph Machine Learning",),
}


def tasks_for_group(group: str | None) -> tuple[str, ...]:
    """Return only the specified group's tasks without reclassifying entries."""
    return AI_TASK_CATALOG.get(group, ())


def validate_catalog_selection(rows: list[dict]) -> None:
    """Validate frontend selections only; leave the backend task contract open."""
    if not rows:
        raise ValueError("Select at least one AI Task.")
    seen = set()
    for index, row in enumerate(rows, start=1):
        group, task = row.get("category"), row.get("task")
        if group not in AI_TASK_CATALOG or task not in tasks_for_group(group):
            raise ValueError(f"AI Task row {index}: select a Group and an AI Task from that Group.")
        pair = (group, task)
        if pair in seen:
            raise ValueError(f"AI Task row {index}: duplicate Group + AI Task combinations are not allowed.")
        seen.add(pair)
