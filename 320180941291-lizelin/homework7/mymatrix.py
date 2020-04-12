class MatrixException(Exception):pass

class Matrix:
#initialize a matrix
	def __init__(self, rows, cols, filling=0):
		self.rows=rows
		self.cols=cols
		self.shape=(rows, cols)
		if isinstance(filling,int) or isinstance(filling,float):
			self.matrix=[[filling for _ in range(cols)] for _ in range(rows)]
		else:
			self.matrix=[[0 for _ in range(cols)] for _ in range(rows)]
	
#get the value of the martrix's element
	def __getitem__(self, index):
		"""
		>>> A=Matrix(3, 3, 2)
		>>> A[1]
		[2, 2, 2]
		>>> A[(0,0)]
		2
		"""
		try:
			if isinstance(index, int):
				return self.matrix[index]
			elif isinstance(index, tuple):
				return self.matrix[index[0]][index[1]]
		except IndexError as e:
			return None
			
#set the value of the martrix's element
	def __setitem__(self, index, value):
		"""
		>>> A=Matrix(3, 3, 1)
		>>> A[0]=2
		>>> A[0]
		[2, 2, 2]
		>>> A[1][1]=3
		>>> A[1][1]
		3
		"""
		try:
			assert isinstance(value,int) or isinstance(value,float)
			if isinstance(index, int):
				self.matrix[index] = [value for _ in range(self.cols)]
			elif isinstance(index, tuple):
				self.matrix[index[0]][index[1]]=value
		except IndexError as e:
			print('the index you inputted is out of range')
		except AssertionError as e:
			print('the value you set must be a number')
	
#return the string of the Matrix
	def __str__(self):
		"""
		>>> M=Matrix(2, 2, 2)
		>>> M[(1,1)]=3
		>>> print(M)
		[[2, 2], [2, 3]]
		"""
		return str(self.matrix)

#add two matrixs together		
	def __add__(self, other):
		"""
		>>> A=Matrix(2, 2, 3)
		>>> B=Matrix(3, 3, 4)
		>>> C=Matrix(3, 3, 1)
		>>> A+B
		the shape of the matrixs don't equal
		>>> print(B+C)
		[[5, 5, 5], [5, 5, 5], [5, 5, 5]]
		"""
		try:
			if isinstance(other, Matrix)==False:
				raise MatrixException
			assert self.shape==other.shape
			M=Matrix(self.rows, self.cols)
			M.matrix=[[self.matrix[i][j]+other.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)]
			return M
		except AssertionError as e:
			print("the shape of the matrixs don't equal")
			return None
		except MatrixException as e:
			return None

#the subtraction of matrixs	
	def __sub__(self, other):
		"""
		>>> A=Matrix(2, 2, 3)
		>>> B=Matrix(3, 3, 4)
		>>> C=Matrix(3, 3, 1)
		>>> C[(1,1)]=2
		>>> A-B
		the shape of the matrixs don't equal
		>>> print(B-C)
		[[3, 3, 3], [3, 2, 3], [3, 3, 3]]
		"""
		try:
			if isinstance(other, Matrix)==False:
				raise MatrixException
			assert self.shape==other.shape
			M=Matrix(self.rows, self.cols)
			M.matrix=[[self.matrix[i][j]-other.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)]
			return M
		except AssertionError as e:
			print("the shape of the matrixs don't equal")
			return None
		except MatrixException as e:
			return None

#the multiplication between matrix and number or among matrixs	
	def __mul__(self, value):
		"""
        >>> A=Matrix(2, 3, 4)
        >>> B=Matrix(3, 4, 2)
        >>> B[(1, 1)]=1
        >>> A[(1, 1)]=3
        >>> print(A)
        [[4, 4, 4], [4, 3, 4]]
        >>> print(B)
        [[2, 2, 2, 2], [2, 1, 2, 2], [2, 2, 2, 2]]
        >>> print(A*B)
        [[24, 20, 24, 24], [22, 19, 22, 22]]
        """
		try:
			if isinstance(value,int) or isinstance(value,float):
				M=Matrix(self.rows, self.cols)
				M.matrix=[[value * self.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)]
				return M
			elif isinstance(value, Matrix):
				assert self.cols==value.rows
				M=Matrix(self.rows, value.cols)
				for i in range(self.rows):
					for j in range(value.cols):
						for k in range(value.rows):
							M.matrix[i][j]+=self.matrix[i][k]*value.matrix[k][j]
				return M
			else:
				raise MatrixException
		except AssertionError as e:
			return None
		except MatrixException as e:
			return None
	
#the transposition of matrix
	def transposition(self):
		"""
		>>> A=Matrix(2, 4, 3)
		>>> A[(0, 1)]=2
		>>> A[(1, 0)]=4
		>>> print(A)
		[[3, 2, 3, 3], [4, 3, 3, 3]]
		>>> print(A.transposition())
		[[3, 4], [2, 3], [3, 3], [3, 3]]
		"""
		M=Matrix(self.cols, self.rows)
		for i in range(self.cols):
			for j in range(self.rows):
				M.matrix[i][j]=self.matrix[j][i]
		return M
					

if __name__ == "__main__":  
	import doctest
	doctest.testmod()
