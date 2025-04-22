import sys
import struct
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

class Position:
    WIDTH = 7
    HEIGHT = 6
    MIN_SCORE = -(WIDTH * HEIGHT) // 2 + 3
    MAX_SCORE = (WIDTH * HEIGHT + 1) // 2 - 3

    def __init__(self, current_position: int = 0, mask: int = 0, moves: int = 0):
        self.current_position = current_position
        self.mask = mask
        self.moves = moves

    @staticmethod
    def bottom_mask_col(col: int) -> int:
        return 1 << (col * (Position.HEIGHT + 1))

    @staticmethod
    def top_mask_col(col: int) -> int:
        return 1 << ((Position.HEIGHT - 1) + col * (Position.HEIGHT + 1))

    @staticmethod
    def column_mask(col: int) -> int:
        return ((1 << Position.HEIGHT) - 1) << (col * (Position.HEIGHT + 1))

    @staticmethod
    def bottom_mask() -> int:
        mask = 0
        for col in range(Position.WIDTH):
            mask |= Position.bottom_mask_col(col)
        return mask

    @staticmethod
    def board_mask() -> int:
        return Position.bottom_mask() * ((1 << Position.HEIGHT) - 1)

    def can_play(self, col: int) -> bool:
        return (self.mask & Position.top_mask_col(col)) == 0

    def play(self, move: int) -> None:
        self.current_position ^= self.mask
        self.mask |= move
        self.moves += 1

    def get_cell(self, col: int, row: int) -> str:
        mask = 1 << (col * (Position.HEIGHT + 1) + row)
        if not (self.mask & mask):
            return '.'
        return 'X' if (self.current_position & mask) else 'O'

    def play_col(self, col: int) -> None:
        self.play((self.mask + Position.bottom_mask_col(col)) & Position.column_mask(col))

    def play_sequence(self, sequence: str):
        """Chỉ chấp nhận chuỗi số từ 1-7, bỏ qua mọi ký tự khác"""
        valid_moves = 0
        for char in sequence:
            if char.isdigit() and '1' <= char <= '7':
                col = int(char) - 1
                if self.can_play(col):
                    self.play_col(col)
                    valid_moves += 1
        return valid_moves

    def is_winning_move(self, col: int) -> bool:
        if not self.can_play(col):
            return False
        
        # Tạo bản sao tạm để kiểm tra
        temp_pos = self.current_position | ((self.mask + self.bottom_mask_col(col)) & self.column_mask(col))
        return self.check_win(temp_pos)

    def get_current_pieces(self):
        return self.current_position if self.moves % 2 == 0 else self.current_position ^ self.mask

    def can_win_next(self) -> bool:
        return self.winning_position() & self.possible()

    def possible(self) -> int:
        return (self.mask + Position.bottom_mask()) & Position.board_mask()

    def winning_position(self) -> int:
        return self.compute_winning_position(self.current_position, self.mask)

    def opponent_winning_position(self) -> int:
        return self.compute_winning_position(self.current_position ^ self.mask, self.mask)

    @staticmethod
    def compute_winning_position(position: int, mask: int) -> int:
        # Vertical
        r = (position << 1) & (position << 2) & (position << 3)

        # Horizontal
        p = (position << (Position.HEIGHT + 1)) & (position << 2 * (Position.HEIGHT + 1))
        r |= p & (position << 3 * (Position.HEIGHT + 1))
        r |= p & (position >> (Position.HEIGHT + 1))
        p = (position >> (Position.HEIGHT + 1)) & (position >> 2 * (Position.HEIGHT + 1))
        r |= p & (position << (Position.HEIGHT + 1))
        r |= p & (position >> 3 * (Position.HEIGHT + 1))

        # Diagonal 1
        p = (position << Position.HEIGHT) & (position << 2 * Position.HEIGHT)
        r |= p & (position << 3 * Position.HEIGHT)
        r |= p & (position >> Position.HEIGHT)
        p = (position >> Position.HEIGHT) & (position >> 2 * Position.HEIGHT)
        r |= p & (position << Position.HEIGHT)
        r |= p & (position >> 3 * Position.HEIGHT)

        # Diagonal 2
        p = (position << (Position.HEIGHT + 2)) & (position << 2 * (Position.HEIGHT + 2))
        r |= p & (position << 3 * (Position.HEIGHT + 2))
        r |= p & (position >> (Position.HEIGHT + 2))
        p = (position >> (Position.HEIGHT + 2)) & (position >> 2 * (Position.HEIGHT + 2))
        r |= p & (position << (Position.HEIGHT + 2))
        r |= p & (position >> 3 * (Position.HEIGHT + 2))

        return r & (Position.board_mask() ^ mask)

    def possible_non_losing_moves(self) -> int:
        assert not self.can_win_next()
        possible_mask = self.possible()
        opponent_win = self.opponent_winning_position()
        forced_moves = possible_mask & opponent_win
        
        if forced_moves:
            if forced_moves & (forced_moves - 1):
                return 0
            possible_mask = forced_moves
        
        return possible_mask & ~(opponent_win >> 1)

    def move_score(self, move: int) -> int:
        score = bin(self.compute_winning_position(self.current_position | move, self.mask)).count('1')
        
        # Ưu tiên đánh cột giữa
        col = (move.bit_length() - 1) // (Position.HEIGHT + 1)
        score += (3 - abs(3 - col)) * 2  # Đi giữa (cột 3) +6, cột 2 và 4 +4, xa hơn ít điểm hơn

        # Ưu tiên tạo thế ép (Fork)
        if self.can_win_next():
            score += 100
        
        return score

    def check_win(self, position: int) -> bool:
        # Kiểm tra 4 hướng
        directions = [
            1,                    # Dọc
            self.HEIGHT + 1,      # Ngang
            self.HEIGHT,          # Chéo /
            self.HEIGHT + 2       # Chéo \
        ]
        
        for delta in directions:
            if (position & (position >> delta) & (position >> (2 * delta)) & (position >> (3 * delta))):
                return True
        return False
    
    def key(self) -> int:
        return self.current_position + self.mask

    def key3(self) -> int:
        key_forward = 0
        for col in range(Position.WIDTH):
            key_forward = self.partial_key3(key_forward, col)

        key_reverse = 0
        for col in range(Position.WIDTH-1, -1, -1):
            key_reverse = self.partial_key3(key_reverse, col)

        return min(key_forward, key_reverse) // 3

    def partial_key3(self, key: int, col: int) -> int:
        pos = 1 << (col * (Position.HEIGHT + 1))
        while pos & self.mask:
            key *= 3
            if pos & self.current_position:
                key += 1
            else:
                key += 2
            pos <<= 1
        key *= 3
        return key

    def __str__(self) -> str:
        board = []
        # Hiển thị số cột
        board.append("  " + "  ".join([str(i+1) for i in range(Position.WIDTH)]))
        
        # Xác định người chơi hiện tại dựa trên số nước đi
        # Nếu số nước đi chẵn: lượt của người chơi (O)
        # Nếu số nước đi lẻ: lượt của AI (X)
        current_player_is_human = (self.moves % 2 == 0)
        
        # Duyệt qua từng ô
        for row in range(Position.HEIGHT-1, -1, -1):
            line = []
            for col in range(Position.WIDTH):
                mask = 1 << (col * (Position.HEIGHT + 1) + row)
                if self.mask & mask:
                    # Nếu ô đã được đánh
                    if self.current_position & mask:
                        # Quân cờ thuộc về người chơi HIỆN TẠI
                        symbol = 'O' if current_player_is_human else 'X'
                    else:
                        # Quân cờ thuộc về người chơi TRƯỚC ĐÓ
                        symbol = 'X' if current_player_is_human else 'O'
                    line.append(symbol)
                else:
                    line.append(".")
            board.append("| " + " | ".join(line) + " |")
        
        # Đường viền dưới
        board.append("+" + "---+" * Position.WIDTH)
        
        return "\n".join(board)