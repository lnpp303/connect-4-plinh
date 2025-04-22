import sys
import struct
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from MoveSorter import MoveSorter
from Position import Position
from Solver import Solver
from TranspositionTable import TranspositionTable

def play_vs_ai(solver: Solver):
    position = Position()
    human_turn = True  # True for human's turn, False for AI's turn
    
    print("Connect Four - Human (O) vs AI (X)")
    print("Nhập số cột (1-7) để chơi\n")
    
    while True:
        print(position)
        
        # Check for draw
        if position.moves == Position.WIDTH * Position.HEIGHT:
            print("Hòa!")
            break
            
        if human_turn:
            # Human's turn
            while True:
                try:
                    col = int(input("Lượt bạn (1-7): ")) - 1
                    if 0 <= col < Position.WIDTH and position.can_play(col):
                        if position.is_winning_move(col):
                            position.play_col(col)
                            print(position)
                            print("Bạn thắng! Xuất sắc!")
                            return
                        position.play_col(col)
                        break
                    print("Cột không hợp lệ hoặc đã đầy!")
                except ValueError:
                    print("Vui lòng nhập số từ 1-7")
        else:
            # AI's turn
            print("\nAI đang suy nghĩ...")
            
            # Analyze the position
            scores = solver.analyze(position)
            
            # Find the best move
            best_col = -1
            best_score = -float('inf')
            
            for col in range(Position.WIDTH):
                if position.can_play(col) and scores[col] > best_score:
                    best_score = scores[col]
                    best_col = col
            
            if best_col != -1:
                print(f"AI chọn cột {best_col + 1}")
                
                if position.is_winning_move(best_col):
                    position.play_col(best_col)
                    print(position)
                    print("AI thắng! Hãy thử lại!")
                    return
                    
                position.play_col(best_col)
            else:
                print("AI không tìm được nước đi hợp lệ!")
                return
        
        human_turn = not human_turn
def main():
    solver = Solver()
    weak = False
    analyze = False
    interactive = False
    args = sys.argv[1:]
    input_from_stdin = False
    
    for arg in args:
        if arg == '-i':
            interactive = True
        elif arg == '-w':
            weak = True
        elif arg == '-a':
            analyze = True
        else:
            input_from_stdin = True

    if interactive:
        play_vs_ai(solver)
        return
    
    if not sys.stdin.isatty() and input_from_stdin:
        for line in sys.stdin:
            line = line.strip()
            if line:
                position = Position()
                try:
                    position.play_sequence(line)
                    if analyze:
                        scores = solver.analyze(position, weak)
                        print(" ".join(map(str, scores)))
                    else:
                        score = solver.solve(position, weak)
                        print(score)
                except ValueError as e:
                    print(f"Lỗi khi xử lý nước đi: {e}")
    else:
        print("Vui lòng nhập chuỗi nước đi hợp lệ hoặc dùng -i để chơi với AI")

if __name__ == "__main__":
    main()
