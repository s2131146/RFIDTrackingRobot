from dataclasses import dataclass
from typing import ClassVar, Set


@dataclass(frozen=True)
class Position:
    LEFT: ClassVar[str] = "L"
    RIGHT: ClassVar[str] = "R"
    CENTER: ClassVar[str] = "C"
    NONE: ClassVar[str] = "X"

    @classmethod
    def invert(cls, pos):
        if pos == cls.LEFT:
            return cls.RIGHT
        if pos == cls.RIGHT:
            return cls.LEFT
        return pos

    @classmethod
    def convert_to_rotate(cls, pos):
        if pos == cls.LEFT or pos == Commands.GO_LEFT:
            return Commands.ROTATE_LEFT
        if pos == cls.RIGHT or pos == Commands.GO_RIGHT:
            return Commands.ROTATE_RIGHT
        if pos == cls.CENTER:
            return Commands.STOP_TEMP
        return pos


@dataclass(frozen=True)
class Commands:
    STOP: ClassVar[str] = "STOP"
    START: ClassVar[str] = "START"
    GO_LEFT: ClassVar[str] = Position.LEFT
    GO_RIGHT: ClassVar[str] = Position.RIGHT
    GO_CENTER: ClassVar[str] = Position.CENTER
    GO_BACK: ClassVar[str] = "B"
    CHECK: ClassVar[str] = "YO"
    STOP_TEMP: ClassVar[str] = "STMP"
    SPD_UP: ClassVar[str] = "UP"
    SPD_DOWN: ClassVar[str] = "DOWN"
    SPD_RESET: ClassVar[str] = "RESET"
    L_SPEED: ClassVar[str] = "LS"
    R_SPEED: ClassVar[str] = "RS"
    SET_DEFAULT_SPEED: ClassVar[str] = "SD"
    ROTATE_RIGHT: ClassVar[str] = "RR"
    ROTATE_LEFT: ClassVar[str] = "RL"
    DETACH_MOTOR: ClassVar[str] = "DETACH"

    DISCONNECT: ClassVar[str] = "DC"
    DEBUG_PID_INIT: ClassVar[str] = "PID"

    LIST_IGNORE_LOG: ClassVar[Set[str]] = {STOP_TEMP, CHECK}
    LIST_ROTATE: ClassVar[Set[str]] = {ROTATE_RIGHT, ROTATE_LEFT}

    @classmethod
    def is_rotate(cls, cmd: str) -> bool:
        return cmd in cls.LIST_ROTATE

    @classmethod
    def is_ignore(cls, cmd: str) -> bool:
        """
        指定されたコマンドが無視リストに含まれているかを判定します。

        Args:
            cmd (str): 判定対象のコマンド文字列。

        Returns:
            bool: コマンドが無視リストに含まれていればTrue、そうでなければFalse。
        """
        return cmd in cls.LIST_IGNORE_LOG

    @classmethod
    def contains(cls, key):
        return any(
            key == value
            for attr, value in cls.__dict__.items()
            if not attr.startswith("__")
        )


@dataclass(frozen=True)
class Cascades:
    FACE = "haarcascade_frontalface_default.xml"
    LOWER_BODY = "haarcascade_lowerbody.xml"
    UPPER_BODY = "haarcascade_upperbody.xml"
    FULL_BODY = "haarcascade_fullbody.xml"
    PROFILE_FACE = "haarcascade_profileface.xml"
