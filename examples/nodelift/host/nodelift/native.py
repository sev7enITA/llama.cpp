"""Small JSON ABI binding to the locked llama.cpp core, no subprocess shell."""
import ctypes
import json
import os
import platform
from pathlib import Path
import sys

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))


def library_path():
    extension = 'dylib' if platform.system() == 'Darwin' else 'so'
    return Path(os.environ.get('NODELIFT_LIBRARY', ROOT/'native/build/host/lib'/('libnodelift.'+extension)))


class Native:
    def __init__(self):
        self.lib = ctypes.CDLL(str(library_path()))
        self.lib.nl_create.restype = ctypes.c_void_p
        self.lib.nl_call.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.nl_call.restype = ctypes.c_void_p
        self.lib.nl_free_string.argtypes = [ctypes.c_void_p]
        self.lib.nl_destroy.argtypes = [ctypes.c_void_p]
        self.lib.nl_cancel.argtypes = [ctypes.c_void_p]
        self.handle = self.lib.nl_create()
        if not self.handle: raise RuntimeError('Native engine allocation failed')

    def call(self, op, **kwargs):
        if not self.handle: raise RuntimeError('Engine closed')
        pointer = self.lib.nl_call(self.handle, json.dumps({'op':op, **kwargs}, allow_nan=False).encode())
        if not pointer: raise RuntimeError('Native response allocation failed')
        try: result = json.loads(ctypes.string_at(pointer))
        finally: self.lib.nl_free_string(pointer)
        if 'error' in result: raise RuntimeError(result['error'])
        return result

    def cancel(self):
        if self.handle: self.lib.nl_cancel(self.handle)

    def close(self):
        if self.handle: self.lib.nl_destroy(self.handle); self.handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
