from typing import List, Dict, Tuple, Optional
from MoveSorter import MoveSorter
from Position import Position
from TranspositionTable import TranspositionTable

class Solver:
    INVALID_MOVE = -1000
    WIN_SCORE = 1000000
    
    def __init__(self):
        self.transposition_table = {}
        self.column_order = [3, 2, 4, 1, 5, 0, 6]  # Center-first ordering
    
    def analyze(self, position: Position) -> List[int]:
        scores = [self.INVALID_MOVE] * Position.WIDTH
        
        # Check each column for valid moves
        for col in range(Position.WIDTH):
            if not position.can_play(col):
                continue
                
            # Check if this is a winning move
            if position.is_winning_move(col):
                scores[col] = self.WIN_SCORE
                return scores  # Immediate win takes priority
        
        # Check if we need to block opponent's win
        for col in range(Position.WIDTH):
            if not position.can_play(col):
                continue
                
            # Create opponent's position by flipping current player
            opponent_pos = Position(position.current_position ^ position.mask, position.mask, position.moves)
            if opponent_pos.is_winning_move(col):
                scores[col] = self.WIN_SCORE - 1  # Very high priority, but less than winning
        
        # For all valid columns, evaluate using minimax
        for col in range(Position.WIDTH):
            if not position.can_play(col):
                continue
                
            if scores[col] == self.INVALID_MOVE:  # If not already scored (win or block)
                # Make the move
                new_pos = Position(position.current_position, position.mask, position.moves)
                new_pos.play_col(col)
                
                # Run minimax with reasonable depth
                scores[col] = -self.minimax(new_pos, 4, -float('inf'), float('inf'))
                print(f"Col {col+1}: score {scores[col]}")
        
        return scores
    
    def minimax(self, position: Position, depth: int, alpha: int, beta: int) -> int:
        """Negamax implementation of minimax with alpha-beta pruning"""
        if position.moves == Position.WIDTH * Position.HEIGHT:
            return 0  # Draw
        
        # Check for immediate win
        for col in range(Position.WIDTH):
            if position.can_play(col) and position.is_winning_move(col):
                return (Position.WIDTH * Position.HEIGHT + 1 - position.moves) // 2
        
        if depth == 0:
            return self.evaluate(position)
        
        max_score = -float('inf')
        
        # Try each column
        for col in self.column_order:
            if position.can_play(col):
                # Make the move
                new_pos = Position(position.current_position, position.mask, position.moves)
                new_pos.play_col(col)
                
                # Recursive evaluation with negation (negamax)
                score = -self.minimax(new_pos, depth - 1, -beta, -alpha)
                
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    break  # Pruning
        
        return max_score
    
    def evaluate(self, position: Position) -> int:
        """Evaluate the position from current player's perspective"""
        score = 0
        
        # Check all possible winning lines
        for row in range(Position.HEIGHT):
            for col in range(Position.WIDTH):
                # Check horizontal lines
                if col <= Position.WIDTH - 4:
                    score += self.evaluate_line(position, col, row, 1, 0)
                
                # Check vertical lines
                if row <= Position.HEIGHT - 4:
                    score += self.evaluate_line(position, col, row, 0, 1)
                
                # Check diagonal lines (/)
                if col <= Position.WIDTH - 4 and row <= Position.HEIGHT - 4:
                    score += self.evaluate_line(position, col, row, 1, 1)
                
                # Check diagonal lines (\)
                if col <= Position.WIDTH - 4 and row >= 3:
                    score += self.evaluate_line(position, col, row, 1, -1)
        
        # Bonus for center control
        for row in range(Position.HEIGHT):
            if position.get_cell(3, row) == 'X':  # Middle column
                score += 3
        
        return score
    
    def evaluate_line(self, position: Position, col: int, row: int, delta_col: int, delta_row: int) -> int:
        """Evaluate a line of 4 cells"""
        # Get the 4 cells in this line
        cells = []
        for i in range(4):
            new_col = col + i * delta_col
            new_row = row + i * delta_row
            
            if 0 <= new_col < Position.WIDTH and 0 <= new_row < Position.HEIGHT:
                cells.append(position.get_cell(new_col, new_row))
            else:
                return 0  # Out of bounds
        
        # Count pieces
        count_x = cells.count('X')
        count_o = cells.count('O')
        count_empty = cells.count('.')
        
        # Cannot have both X and O in a winning line
        if count_x > 0 and count_o > 0:
            return 0
        
        # Current player's perspective
        current_player = 'X' if position.moves % 2 == 1 else 'O'
        opponent = 'O' if current_player == 'X' else 'X'
        
        # Current player's pieces
        if count_x > 0 and current_player == 'X':
            if count_x == 3 and count_empty == 1:
                return 50  # Three in a row with an empty spot
            elif count_x == 2 and count_empty == 2:
                return 10  # Two in a row with two empty spots
            elif count_x == 1 and count_empty == 3:
                return 1   # One piece with three empty spots
        
        # Opponent's pieces - negative scores
        if count_o > 0 and current_player == 'X':
            if count_o == 3 and count_empty == 1:
                return -100  # Block opponent's three in a row (high priority)
            elif count_o == 2 and count_empty == 2:
                return -10   # Block potential development
        
        # Same logic for when current player is O
        if count_o > 0 and current_player == 'O':
            if count_o == 3 and count_empty == 1:
                return 50
            elif count_o == 2 and count_empty == 2:
                return 10
            elif count_o == 1 and count_empty == 3:
                return 1
        
        if count_x > 0 and current_player == 'O':
            if count_x == 3 and count_empty == 1:
                return -100
            elif count_x == 2 and count_empty == 2:
                return -10
        
        return 0