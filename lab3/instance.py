from dataclasses import dataclass

@dataclass
class Instance:
    text: list[str]
    label: str