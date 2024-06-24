from dataclasses import dataclass

@dataclass(frozen=True)
class Commands():
    STOP = "STOP"
    START = "START"
    SET_SPD_LEFT = "L"
    SET_SPD_RIGHT = "R"
    CHECK = "YO"

    @classmethod
    def contains(cls, key):
        return any(key == value for attr, value in cls.__dict__.items() if not attr.startswith('__'))