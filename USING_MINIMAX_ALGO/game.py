from math import inf as infinity
from random import choice


board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

def no_of_zeroes(state):
    count = 0

    for i in state:
        for j in i:
            if j==0:
                count+=1
    return count


def empty_cells(state):
    cells = [] 

    for i in range(3):
        for j in range(3):
            if state[i][j]==0:
                cells.append((i,j))
    return cells

def set_move(state, x, y, player):

    if state[x][y] == 0:
        state[x][y] = player
        return True
    else:
        return False



def show_board(state):

    sym = [ ' ', 'O', 'X']
    str_line = '---------------'

    print('\n' + str_line)
    for i in state:
        for j in i:
            print(f'| {sym[j]} |', end='')
        print('\n' + str_line)

def wins(state, n):
    j=0
    for i in range(3):
        if ( state[i][j],  state[i][j+1], state[i][j+2] ) ==( n,n,n ) :
            return True
    for i in range(3):
        if (state[j][i],  state[j+1][i], state[j+2][i]) == (n,n,n):
            return True
    if (state[j][j], state[j+1][j+1], state[j+2][j+2]) == (n,n,n):
        return True
    if (state[j][j+2], state[j+1][j+1], state[j+2][j]) == (n,n,n):
        return True
    return False

def evaluate(state):

    if wins(state, 2):
        score = +1
    elif wins(state, 1):
        score = -1
    else:
        score = 0

    return score

def game_over(state):
    return wins(state, 1) or wins(state, 2)


def minimax(state, depth, player):
    if player == 2:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == 2:
            if score[2] > best[2]:
                best = score  
        else:
            if score[2] < best[2]:
                best = score  
    return best


def AI_turn():
    depth = no_of_zeroes(board)
    if depth == 0 or game_over(board):
        return
    
    show_board(board)
    
    move = minimax(board, no_of_zeroes(board), 2)
    set_move(board, move[0], move[1], 2)


def H_turn():
    depth = no_of_zeroes(board)
    if depth == 0 or game_over(board):
        return

    show_board(board)

    move = -1
    while move < 1 or move > 9:
        move = int(input('Use numpad (1..9): '))
        co_ord = ( (move-1)//3, (move-1)%3 ) 
        can_move = set_move(board, co_ord[0], co_ord[1], 1)

        if not can_move:
            print('Bad move')
            move = -1



def main():
    human = 'O'
    ai = 'X'

    while no_of_zeroes(board) >0 and not game_over(board):
        H_turn()
        AI_turn()

    if wins(board, 1):
        show_board(board)
        print("You Wins!!")

    elif wins(board, 2):
        show_board(board)
        print("Computer Wins!!")

    else:
        show_board(board)
        print("Draw")

    exit()


main()
