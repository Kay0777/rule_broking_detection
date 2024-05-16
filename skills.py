from collections import deque

NUMBERS = '12345678'
LETTERS = 'abcdefgh'


CHESS = [[f'{i}{j}' for j in LETTERS] for i in NUMBERS]

for row in CHESS:
    print(row)


def combinations(i, j) -> tuple[int, int]:
    return (
        (i - 2, j - 1),
        (i - 2, j + 1),
        (i - 1, j - 2),
        (i - 1, j + 2),
        (i + 2, j - 1),
        (i + 2, j + 1),
        (i + 1, j - 2),
        (i + 1, j + 2)
    )


def move_knight(current_position: str):
    global NUMBERS, LETTERS

    i: int = list(NUMBERS).index(current_position[0])
    j: int = list(LETTERS).index(current_position[1])

    return [
        CHESS[m][n]
        for m, n in combinations(i=i, j=j)
        if 0 < m and m < 8 and 0 < n and n < 8
    ]


def find_min_move_count(start_position: str, exit_position: str) -> int:
    seens: set = set()
    d = deque()
    d.append(start_position)

    counter = 1
    while d:
        pos = d.pop()
        for move in move_knight(current_position=pos):
            if move == exit_position:
                return counter

            if move not in seens:
                seens.add(move)
                d.append(move)
        counter += 1


find_min_move_count(start_position='5d')
