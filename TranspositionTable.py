import sys
import struct
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

class TranspositionTable:
    def __init__(self, key_size: int = 8, value_size: int = 4, log_size: int = 24):
        self.size = self.next_prime(1 << log_size)
        self.keys = [0] * self.size
        self.values = [0] * self.size
        self.key_size = key_size
        self.value_size = value_size
        self.max_value = (1 << (8 * value_size)) // 2 - 1

    @staticmethod
    def is_prime(n: int) -> bool:
        """Kiểm tra số nguyên tố hiệu quả"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        w = 2
        while i * i <= n:
            if n % i == 0:
                return False
            i += w
            w = 6 - w  # Chuyển đổi giữa bước 2 và 4 (5->7->11->13...)
        return True

    @staticmethod
    def next_prime(n: int) -> int:
        """Tìm số nguyên tố tiếp theo"""
        if n < 2:
            return 2
        if n % 2 == 0:
            n += 1
        while not TranspositionTable.is_prime(n):
            n += 2
        return n

    def index(self, key: int) -> int:
        """Tính index trong bảng hash"""
        return key % self.size

    def put(self, key: int, value: int) -> None:
        """Thêm cặp key-value vào bảng"""
        if abs(value) > self.max_value:
            raise ValueError(f"Value {value} exceeds maximum size {self.max_value}")
        
        idx = self.index(key)
        original_idx = idx
        while True:
            if self.keys[idx] == 0 or self.keys[idx] == key:
                self.keys[idx] = key
                self.values[idx] = value
                return
            idx = (idx + 1) % self.size
            if idx == original_idx:  # Bảng đã đầy
                raise RuntimeError("Transposition table is full")

    def get(self, key: int) -> Optional[int]:
        """Lấy giá trị từ key"""
        idx = self.index(key)
        original_idx = idx
        while self.keys[idx] != 0:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.size
            if idx == original_idx:
                break
        return None

    def clear(self):
        """Xóa toàn bộ nội dung bảng"""
        self.keys = [0] * self.size
        self.values = [0] * self.size