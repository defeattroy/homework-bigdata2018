#320180941011-huyelong Thursday, March 19, 2020
class MyMatrix(object):
	"""
	:parameter: a: MyMatrix, b: MyMatrix
	:function:
	addition: a.add(b)
	subtraction: a.sub(b)
	multiplication: a.mul(b)
	dot product: a.dot(b)
	transpose: a.trans(b)
	"""
	def __init__(self, list_1):
		"""
		matrix can only be created by a list
		"""
		self.matrix = list_1
		self.shape = (len(list_1),len(list_1[0]))
		self.row = self.shape[0]
		self.column = self.shape[1]

	def zeroValueMatrix(self, shape):
		"""
		create a zero matrix to store the results
		"""
		zvMat = []
		for i in range(shape[0]):
			zvMat.append([])
			for j in range(shape[1]):
				zvMat[i].append(0)

		zero_value_matrix = MyMatrix(zvMat)

		return zero_value_matrix

	def add(self, mat2):
		"""
		matrix addition
		"""
		assert isinstance(mat2, MyMatrix), "The second Matrix is not MyMatrix class"
		assert mat2.shape == self.shape, "Two matrix dimensions do not match, cannot be added"

		result = self.zeroValueMatrix(self.shape)

		for i in range(self.row):
			for j in range(self.column):
				result.matrix[i][j] = self.matrix[i][j] + mat2.matrix[i][j]

		return result

	def sub(self, mat2):
		"""
		matrix subtraction
		"""
		assert isinstance(mat2, MyMatrix), "The second Matrix is not MyMatrix class"
		assert mat2.shape == self.shape, "Two matrix dimensions do not match, cannot be added"

		result = self.zeroValueMatrix(self.shape)

		for i in range(self.row):
			for j in range(self.column):
				result.matrix[i][j] = self.matrix[i][j] - mat2.matrix[i][j]

		return result

	def mul(self, mat2):
		"""
		matrix multiplication
		"""
		assert isinstance(mat2, MyMatrix), "The second Matrix is not MyMatrix class"
		assert self.shape[1] == mat2.shape[0], "The number of columns in the first matrix does not match the number of rows in the second matrix"

		shape = (self.shape[0], mat2.shape[1])
		result = self.zeroValueMatrix(shape)

		for i in range(self.shape[0]):
			for j in range(mat2.shape[1]):
				number = 0
				for k in range(self.shape[1]):
					number += self.matrix[i][k] * mat2.matrix[k][j]
				result.matrix[i][j] = number

		return result

	def dot(self,mat2):
		"""
		matrix dot
		"""
		assert isinstance(mat2, MyMatrix), "The second Matrix is not MyMatrix class"
		assert mat2.shape == self.shape, "Two matrix dimensions do not match"

		result = self.zeroValueMatrix(self.shape)

		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				result.matrix[i][j] = self.matrix[i][j] * mat2.matrix[i][j]

		return result

	def trans(self):
		"""
		matrix transpose
		"""

		self = (list(zip(*self.matrix)))

		return self


if __name__ == "__main__":
	import doctest
	doctest.testmod(verbose = True)
	
	#mat1: 2*3 matrix
	mat1 = MyMatrix([[1,2,3],[4,5,6]])
	#mat2: 3*2 matrix
	mat2 = MyMatrix([[4,5],[6,7],[8,9]])
	#mat3: 2*3 matrix
	mat3 = MyMatrix([[4,5,6],[7,8,9]])

	#test
	print("mat1:", mat1.matrix)
	print("mat2:", mat2.matrix)
	print("mat3:", mat3.matrix)
	addition = mat1.add(mat3)
	print(addition.matrix)
	subtraction = mat1.sub(mat3)
	print(subtraction.matrix)
	multiplication = mat1.mul(mat2)
	print(multiplication.matrix)
	dot_product = mat1.dot(mat3)
	print(dot_product.matrix)
	transpose = mat1.trans()
	print(transpose)