from homework7.Vector import Vector
from homework7.mymatrix import Matrix


if __name__ == "__main__":

    matrix = Matrix([[2, 6], [4, 7]])
    matrix2 = Matrix([[6, 3], [8, 9]])
    T = Matrix([[2.3, 0], [0, 2]])
    p = Vector([3, 5])
    P = Matrix([[0, 4, 5], [0, 0, 3]])
    print("matrix=",matrix)
    print("matrix.shape = {}".format(matrix.shape()))
    print("matrix.size = {}".format(matrix.size()))
    print("len(matrix) = {}".format(len(matrix)))
    print("matrix[0][0] = {}".format(matrix[0, 0]))

    print("matrix2=",matrix2)
    print("add: {}".format(matrix + matrix2))
    print("subtract: {}".format(matrix - matrix2))
    print("scalar-mul: {}".format(2 * matrix))
    print("scalar-mul: {}".format(matrix * 2))
    print("null_2_3: {}".format(Matrix.null(2, 3)))

	#Test matrix times vector
    print("p=",p)
    print("T=",T)
    print("T.dot(p) = {}".format(T.dot(p)))
	#Test matrix multiplied by matrix
    print("P=",P)
    print("T.dot(P) = {}".format(T.dot(P)))
	#Verify the commutative property of two matrices
    print("A.dot(B) = {}".format(matrix.dot(matrix2)))
    print("B.dot(A) = {}".format(matrix2.dot(matrix)))
