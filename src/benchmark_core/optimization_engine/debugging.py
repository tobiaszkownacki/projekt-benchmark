import multiprocessing as mp
import os
from multiprocessing.connection import Connection

import torch


class VerbosePipe:
    def __init__(self, conn: Connection):
        self._conn = conn

    def _tag(self) -> str:
        return f"[{mp.current_process().name} pid={os.getpid()}]"

    def send(self, obj):
        print(f"{self._tag()} PIPE send -> {obj}", flush=True)
        return self._conn.send(obj)

    def recv(self):
        obj = self._conn.recv()
        print(f"{self._tag()} PIPE recv <- {obj}", flush=True)
        return obj

    def close(self):
        return self._conn.close()


class VerboseTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor, key: str):
        instance = torch.Tensor._make_subclass(cls, data, data.requires_grad)
        instance._key = key
        return instance

    def _tag(self) -> str:
        return f"[{mp.current_process().name} pid={os.getpid()}]"

    def copy_(self, src, non_blocking: bool = False):
        res = super().copy_(src, non_blocking=non_blocking)
        print(
            f"{self._tag()} SHARED[{self._key!r}].copy_ -> {self.tolist()}",
            flush=True,
        )
        return res

    def __setitem__(self, idx, val):
        super().__setitem__(idx, val)
        print(
            f"{self._tag()} SHARED[{self._key!r}][{idx}] = {val} -> {self.tolist()}",
            flush=True,
        )


class VerboseSharedDict:
    def __init__(self, data: dict):
        self._data = data

    def _tag(self) -> str:
        return f"[{mp.current_process().name} pid={os.getpid()}]"

    def __getitem__(self, key: str):
        val = self._data[key]
        if isinstance(val, torch.Tensor):
            return VerboseTensor(val, key)
        return val

    def __setitem__(self, key: str, value):
        if key in self._data and isinstance(self._data[key], torch.Tensor):
            if isinstance(value, torch.Tensor):
                self._data[key].copy_(value)
            else:
                self._data[key][0] = value
        else:
            self._data[key] = value
        print(f"{self._tag()} SHARED[{key!r}] = {value}", flush=True)

    def clear(self):
        self._data.clear()
