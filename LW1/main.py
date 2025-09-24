import numpy as np

def generate_matrix(rows, cols):
    matrix = np.arange(rows * cols, 0, -1).reshape(rows, cols, order='F')
    return matrix

def main() -> None:
    matrix = generate_matrix(5, 5)
    print(matrix)
    pass


if __name__ == '__main__':
    main()
