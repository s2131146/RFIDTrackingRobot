from dataclasses import dataclass
from typing import ClassVar, Set


@dataclass(frozen=True)
class Commands:
    STOP: ClassVar[str] = "STOP"
    START: ClassVar[str] = "START"
    GO_LEFT: ClassVar[str] = "L"
    GO_RIGHT: ClassVar[str] = "R"
    GO_CENTER: ClassVar[str] = "C"
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

    DEBUG_PID_INIT: ClassVar[str] = "PID"

    IGNORE_LOG: ClassVar[Set[str]] = {STOP_TEMP, CHECK}

    @classmethod
    def is_ignore(cls, cmd: str) -> bool:
        """
        指定されたコマンドが無視リストに含まれているかを判定します。

        Args:
            cmd (str): 判定対象のコマンド文字列。

        Returns:
            bool: コマンドが無視リストに含まれていればTrue、そうでなければFalse。
        """
        return cmd in cls.IGNORE_LOG

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
