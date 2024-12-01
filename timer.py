import math
import time
from typing import List, Tuple


class timer:
    _start_time_list: List[Tuple[str, float, bool]] = []

    def register(self, name: str, show: bool = True, update: bool = False):
        if not self.has(name):
            self._start_time_list.append((name, time.time(), show))
        elif update:
            self.update(name)

    def update(self, name: str):
        if self.has(name):
            c = self.get(name)
            index = self._start_time_list.index(c)
            self._start_time_list[index] = [name, time.time(), c[2]]

    def has(self, name: str) -> bool:
        return any(t[0] == name for t in self._start_time_list)

    def get(self, name: str):
        if self.has(name):
            return [t for t in self._start_time_list if t[0] == name][0]
        return None

    def get_start_time(self, name: str):
        if self.has(name):
            return self.get(name)[1]
        return None

    def is_show(self, name: str):
        if self.has(name):
            return self.get(name)[2]
        return False

    def remove(self, name: str):
        if self.has(name):
            self._start_time_list.remove(self.get(name))

    def yet(self, name: str, until: float, remove: bool = False) -> bool:
        if self.has(name):
            ret = time.time() - self.get_start_time(name) < until
            if remove and not ret:
                self.remove(name)
            return ret
        else:
            return False

    def passed(
        self, name: str, sec: float, update: bool = False, remove: bool = False
    ) -> bool:
        if self.has(name):
            ret = time.time() - self.get_start_time(name) >= sec
            if update and ret:
                self.update(name)
            if remove and ret:
                self.remove(name)
            return ret
        else:
            return False

    def get_elapsed(self, name: str) -> float:
        if self.has(name):
            return time.time() - self.get_start_time(name)
        else:
            self.register(name)
            return 0

    def get_all(self) -> List[Tuple[str, float]]:
        return [
            (name, time.time() - start_time)
            for name, start_time, show in self._start_time_list
            if show
        ]

    @classmethod
    def elapsed_str(cls, start):
        return "{}ms".format(math.floor((time.time() - start) * 1000))
