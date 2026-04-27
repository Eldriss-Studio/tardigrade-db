from typing import Protocol


class PrefixFormat(Protocol):
    def format(self, memories: list[dict]) -> str: ...


TIER_NAMES = {0: "Draft", 1: "Validated", 2: "Core"}


class BulletListFormat:
    """Simple bullet list — clean, model-agnostic."""

    def format(self, memories: list[dict]) -> str:
        if not memories:
            return ""
        lines = ["Memory context:"]
        for m in memories:
            text = m["text"].replace("\n", " ")
            lines.append(f"- {text}")
        return "\n".join(lines)


class TierAnnotatedFormat:
    """Annotates each fact with its governance tier."""

    def format(self, memories: list[dict]) -> str:
        if not memories:
            return ""
        lines = ["Memory context:"]
        for m in memories:
            text = m["text"].replace("\n", " ")
            tier = TIER_NAMES.get(m["tier"], "Unknown")
            lines.append(f"- [{tier}] {text}")
        return "\n".join(lines)
