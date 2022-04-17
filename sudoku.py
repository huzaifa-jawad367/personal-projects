def find_next_empty(puzzle):

    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == -1:
                return r,c
    
    return None,None

def is_valid(puzzle, guess, row, col):
    # check the row
    row_vals = puzzle[row]
    if guess in row_vals:
        return False
    
    # check te coloumn
    col_vals = [puzzle[i][col] for i in range(9)]
    if guess in col_vals:
        return False
    
    # check the box
    row_start = (row//3)*3
    col_start = (col//3)*3

    for r in range(row_start, row_start+3):
        for c in range(col_start, col_start+3):
            if puzzle[r][c] == guess:
                return False

    return True
            
def sudoku_solver(puzzle):
    row, col = find_next_empty(puzzle)

    if row == None:
        return True

    for guess in range(1, 10):
        if is_valid(puzzle, guess, row, col):
            puzzle[row][col] = guess

            if sudoku_solver(puzzle):
                return True

        # if not valid OR guess does not solve this then back track and try a new number
        puzzle[row][col] = -1

    # puzzle is unsolvable
    return False

example_board = [
    [3, 9, -1,    -1, 5, -1,    -1, -1, -1],
    [-1, -1, -1,    2, -1, -1,    -1, -1, 5],
    [-1, -1, -1,   7, 1, 9,    -1, 8, -1],

    [-1, 5, -1,    -1, 6, 8,    -1, -1, -1],
    [2, -1, 6,    -1, -1, 3,    -1, -1, -1],
    [-1, -1, -1,    -1, -1, -1,    -1, -1, 4],

    [5, -1, -1,    -1, -1, -1,    -1, -1, -1],
    [6, 7, -1,    1, -1, 5,    -1, 4, -1],
    [1, -1, 9,    -1, -1, -1,    2, -1, -1]
]

print(sudoku_solver(example_board))
print(example_board)