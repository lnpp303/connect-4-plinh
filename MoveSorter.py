import sys
import struct
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from Position import Position
from TranspositionTable import TranspositionTable

class MoveSorter:
    def __init__(self):
        self.size = 0
        self.entries = [{'move': 0, 'score': 0} for _ in range(Position.WIDTH)]

    def add(self, move: int, score: int) -> None:
        pos = self.size
        self.size += 1
        while pos > 0 and self.entries[pos-1]['score'] > score:
            self.entries[pos] = self.entries[pos-1]
            pos -= 1
        self.entries[pos]['move'] = move
        self.entries[pos]['score'] = score

    def get_next(self) -> int:
        if self.size > 0:
            self.size -= 1
            return self.entries[self.size]['move']
        return 0

    def reset(self) -> None:
        self.size = 0