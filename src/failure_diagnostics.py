"""Conservative error messages for research diagnostics, never raw SDK payloads."""

import os
import re

from src import config


_WITHHELD = "Exception details withheld because they may contain sensitive request or response data."
_STATIC_MESSAGES = {
    "OPENAI_API_KEY is required for contextual generation.",
    "Contextual risk inference requires retrieved NIST evidence.",
    "Treatment retrieval requires contextual risks.",
    "Requirement derivation requires contextual risks.",
    "Requirement derivation requires treatment evidence.",
    "Duplicate retrieved NIST evidence IDs are not allowed.",
    "Duplicate treatment NIST evidence IDs are not allowed.",
    "Require 1 <= top_k <= candidate_k <= 72.",
    "The retrieval query must be a non-empty string.",
    "Reranker must return one finite score per candidate.",
    "Requirement AI tasks must be associated with its addressed contextual risks.",
    "Requirement statement must specify a system-level control, not an organizational recommendation.",
    "The NIST Playbook must be a JSON array.",
    "NIST Playbook titles must be unique.",
}


def sanitize_error_message(error: Exception) -> str:
    """Allow local validation reasons; withhold arbitrary text and unsafe ID values."""
    message = str(error)
    for key in (config.OPENAI_API_KEY, os.getenv("OPENAI_API_KEY")):
        if key and key in message:
            return _WITHHELD
    if type(error) is not ValueError:
        return _WITHHELD
    if message in _STATIC_MESSAGES:
        return message
    if re.fullmatch(
        r"Duplicate (?:risk|evidence|requirement|treatment evidence) IDs are not allowed\.|"
        r"(?:AI task|NIST evidence|contextual risk) references must not be empty\.|"
        r"Duplicate (?:AI task|NIST evidence|contextual risk) references are not allowed\.",
        message,
    ):
        return message
    prefixes = (
        "Unknown NIST evidence references: ", "Unknown AI task references: ",
        "Unknown contextual risk references: ", "Unknown NIST evidence ID: ",
        "Invalid or duplicate NIST candidate identity: ", "NIST candidate content mismatch: ",
        "Evidence does not match authoritative record: ",
    )
    for prefix in prefixes:
        if message.startswith(prefix):
            values = message[len(prefix):]
            # Only bounded canonical reference syntax is safe to echo from generated text.
            if len(values) <= 512 and all(re.fullmatch(
                r"(?:GOVERN|MAP|MEASURE|MANAGE) [0-9]{1,3}\.[0-9]{1,3}|(?:TASK|R|REQ)-[0-9]{2,6}",
                value,
            ) for value in values.split(", ")):
                return message
            return prefix + "[redacted identifier]"
    return _WITHHELD
