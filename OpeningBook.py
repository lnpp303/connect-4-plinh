import sys
import struct
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from Position import Position
from TranspositionTable import TranspositionTable

class OpeningBook:
    def __init__(self, width: int = Position.WIDTH, height: int = Position.HEIGHT):
        self.width = width
        self.height = height
        self.depth = -1
        self.table = None

    def load(self, filename: str) -> None:
        try:
            with open(filename, 'rb') as f:
                _width = struct.unpack('B', f.read(1))[0]
                _height = struct.unpack('B', f.read(1))[0]
                _depth = struct.unpack('B', f.read(1))[0]
                key_bytes = struct.unpack('B', f.read(1))[0]
                value_bytes = struct.unpack('B', f.read(1))[0]
                log_size = struct.unpack('B', f.read(1))[0]

                if (_width != self.width or _height != self.height or 
                    _depth > self.width * self.height or key_bytes > 8 or value_bytes != 1):
                    print("Invalid opening book format")
                    return

                self.table = TranspositionTable(key_bytes, value_bytes, log_size)
                f.readinto(self.table.keys)
                f.readinto(self.table.values)
                self.depth = _depth
                print(f"Loaded opening book with depth {self.depth}")
        except Exception as e:
            print(f"Error loading opening book: {e}")
            self.table = None
            self.depth = -1

    def get(self, position: Position) -> int:
        if position.moves > self.depth or not self.table:
            return 0
        return self.table.get(position.key3())