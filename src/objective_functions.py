import numpy as np

class ObjectiveFunctions:
    def __init__(self, dimension):
        self.functions = {
            'ackley': Ackley(dimension),
            'sphere': Sphere(dimension),
            'sum_square': Sum_square(dimension),
            'levy': Levy(dimension),
            'rosenbrock': Rosenbrock(dimension)
        }

    def evaluate(self, name, x):
        name_key = name.lower()
        if name_key not in self.functions:
            raise ValueError(f"Unknown function {name_key}")
        
        return self.functions[name_key].real(x)
    
class Ackley:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        term1 = -0.2 * np.sqrt(np.sum(x ** 2, axis=1) / self.dimension)
        term2 = np.sum(np.cos(2 * np.pi * x), axis=1) / self.dimension
        return -(-20 * np.exp(term1) - np.exp(term2) + 20 + np.e)

class Sum_square:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        coefficients = np.arange(1, self.dimension + 1)
        return -np.sum(coefficients * x**2, axis=1)

class Sphere:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        return -np.sum(x**2, axis=1)

class Levy:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        w = 1 + (x - 1) / 4
        term1 = (np.sin(np.pi * w[:, 0])) ** 2
        wi = w[:, :-1]
        term_middle = np.sum((wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2), axis=1)
        term_last = (w[:, -1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[:, -1])) ** 2)
        return -(term1 + term_middle + term_last)

class Rosenbrock:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        sum_terms = 100 * (x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1)**2
        return -(np.sum(sum_terms, axis=1))

