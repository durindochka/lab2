import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve,hilbert

def check_if_matrix_invertible(matrix: np.ndarray) -> bool:  # type: ignore
    if np.linalg.det(matrix) != 0:
        return True
    if np.linalg.det(matrix) == 0:
        return False
    
def check_if_matrix_main_minors_invertible(matrix: np.ndarray, number_of_rows_and_columns: int):
    calculate = 0
    while number_of_rows_and_columns > calculate:
        data_without_column = np.delete(matrix, calculate, 1)
        data_without_row = np.delete(data_without_column, calculate, 0)
        if np.linalg.det(data_without_row) != 0:
            calculate += 1
            print("Minor " + str(calculate) + " is invertible")
        else:
            break
  

def generate_vector_b(matrix, number_of_rows_and_columns, result):
        number = number_of_rows_and_columns
        x = result
        to_calculate = 0
        b = np.empty(number)
        element_b = []
        while to_calculate < number:
            element_b = x*(matrix[to_calculate])
            sum_element_b = np.sum(element_b)
            b[to_calculate] = sum_element_b
            to_calculate += 1   
        return b

def gauss_method(matrix, vector_b):
    b = vector_b
    P, L, U = lu(matrix)
    np.allclose(L @ U, matrix)
    LU, p = lu_factor(matrix)
    lu_solve((LU, p), b)
    solve = np.linalg.solve(matrix, b)
    print("\nHilbert method: \n" + str(solve))


def is_diagonal_matrix(matrix, number_rows_and_columns:int) -> bool:
    ''' true if is diagonal'''
    rows = 0
    columns = 0
    
    for_do_while:bool = True
    
    while for_do_while:
        if abs(matrix[rows, columns]) > np.sum(np.absolute(np.delete((matrix[rows]), columns))):
            rows += 1
            columns += 1
        if number_rows_and_columns - 1 == rows and number_rows_and_columns- 1 == columns:
            return True
        else:    
            for_do_while = False
        if number_rows_and_columns <= rows:
            for_do_while = False
    return False

def transform_to_diagonal(matrix, number_rows_and_columns):
        number_to_calculate = 0
        while number_rows_and_columns > number_to_calculate:
            matrix[ number_to_calculate, number_to_calculate] = 0
            matrix[matrix == matrix[number_to_calculate, number_to_calculate]] = np.sum(np.absolute(matrix[number_to_calculate]))*1.1
            number_to_calculate += 1
           
        return matrix   



def jacobi_method(matrix, number_of_iterations, vector_b, guess):
    b = vector_b  
    D = np.diag(np.diag(matrix))
    R = matrix - D
    x = guess
    for i in range (number_of_iterations):
        x = np.dot(np.linalg.inv(D), (b - np.dot(R, x)))
    return x

def seidel_method(matrix, vector_b, number_of_iterations, number_rows_and_columns):
    b = vector_b
    n = number_of_iterations
    x = np.ones(number_rows_and_columns)
    L = np.tril(matrix)
    U = matrix - L
    for i in range(n):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
    return x



if __name__ == '__main__':
    print("Input number of rows and columns of hilbert matrix:")   
    number = int(input())
    hilbert_matrix =np.array(hilbert(number)) 
    print(hilbert_matrix)
    
        
    print("Input number of rows and columns of random matrix:")
    number_2 = int(input())
    random_matrix = np.random.randint(10, size=(number_2, number_2))
    print(random_matrix)
    
    x = lambda a : np.arange(1, a + 1)
    
    print("\nCheck the the hilbert matrix")
    if check_if_matrix_invertible(hilbert_matrix):
        check_if_matrix_main_minors_invertible(hilbert_matrix, number)
        print("\nIt is suitable for this method matrix, continue")
        gauss_method(hilbert_matrix, generate_vector_b(hilbert_matrix, number, x(number)))
    else:
        print("\nIt isn't suitable for this method matrix, continue")
        
    print("\nCheck the the random matrix")
    if check_if_matrix_invertible(random_matrix):
        check_if_matrix_main_minors_invertible(random_matrix, number_2)
        print("\nIt is suitable for this method matrix, continue")
        gauss_method(random_matrix, generate_vector_b(random_matrix, number_2, x(number_2)))
    else:
        print("\nIt isn't suitable for this method matrix, continue")
        
  
    
    print("\nCheck the the hilbert matrix")
    if not is_diagonal_matrix(hilbert_matrix, number):
        transformed_hilbert_matrix = transform_to_diagonal(hilbert_matrix, number)
        hilbert_matrix = transformed_hilbert_matrix
        print("It isn't a diagonal matrix, so change it so that will be:")
        print(transformed_hilbert_matrix)
        
    
    print("\nCheck the the random matrix")
    if not is_diagonal_matrix(random_matrix, number_2):
        transformed_random_matrix = transform_to_diagonal(random_matrix, number_2)
        random_matrix = transformed_random_matrix
        print("It isn't a diagonal matrix, so change it so that will be:")
        print(transformed_random_matrix)    
    else:
        print("It is a diagonal matrix, so we don't change it")



    print("\nJacobi_method:")
    print(jacobi_method(hilbert_matrix, 100, generate_vector_b(hilbert_matrix, number, x(number)), np.ones(number)))
        
    print("\nJacobi_method:")
    print(jacobi_method(random_matrix, 100, generate_vector_b(random_matrix, number_2, x(number_2)), np.ones(number_2)))
    
    print("\nSeidel_method:")
    print(seidel_method(hilbert_matrix,generate_vector_b(hilbert_matrix, number, x(number)), 10, number))
    
    print("\nSeidel_method:")
    print(seidel_method(random_matrix,generate_vector_b(random_matrix, number_2, x(number_2)), 10, number_2))
   



















