import numpy as np

# --- Search domain per function -------------------------------------------------
# Functions not listed here inherit the experiment bounds (the LOFO / earlbo /
# pure_bo pipelines all run on [-1, 1]^d), which is fine for the five smooth /
# valley benchmarks. The three harder functions below MUST use their canonical
# domains: on a small symmetric box Schwefel's optimum (x* ~= 420.97) is entirely
# outside the search region, and Rastrigin / Michalewicz collapse to
# unrepresentative sub-instances. These override whatever bounds are passed in.
CANONICAL_DOMAINS = {
    "rastrigin": (-5.12, 5.12),
    "schwefel": (-500.0, 500.0),
    "michalewicz": (0.0, float(np.pi)),
}

# Best attainable value of real(x) over the search domain, used as the regret
# reference: regret = REFERENCE_OPTIMUM - max(y seen so far).
#   - ackley/sphere/sum_square/levy/rosenbrock/rastrigin: real == -(f_standard),
#     f_standard has minimum 0 at an interior point, so real attains 0.
#   - schwefel: on its full [-500, 500] domain f_standard reaches ~0 at
#     x* ~= 420.97, so real attains ~0 (residual < 1.3e-4 * d from the 418.9829
#     constant); treated as 0.
#   - michalewicz (m=10): real == -(f_standard); f_standard's known minimum is
#     negative and dimension-dependent, so real attains |f_min|. Values below are
#     the accepted optima for the dimensions this project runs (Molga & Smutnicki;
#     Vanaret et al. 2020 for d=10; d=3 is the commonly cited -2.7603947).
_MICHALEWICZ_OPTIMA = {
    2: 1.8013034,
    3: 2.7603947,
    5: 4.687658,
    10: 9.6601517,
}


def search_domain(name):
    """(lower, upper) canonical bounds for `name`, or None to use config bounds."""
    return CANONICAL_DOMAINS.get(name.lower())


def reference_optimum(name, dimension):
    """max of real(x) over the search domain (the regret reference)."""
    key = name.lower()
    if key == "michalewicz":
        try:
            return _MICHALEWICZ_OPTIMA[dimension]
        except KeyError:
            raise KeyError(
                f"No Michalewicz optimum tabulated for dimension {dimension}; "
                "add it to _MICHALEWICZ_OPTIMA in objective_functions.py."
            )
    return 0.0


class ObjectiveFunctions:
    def __init__(self, dimension):
        self.functions = {
            'ackley': Ackley(dimension),
            'sphere': Sphere(dimension),
            'sum_square': Sum_square(dimension),
            'levy': Levy(dimension),
            'rosenbrock': Rosenbrock(dimension),
            'rastrigin': Rastrigin(dimension),
            'schwefel': Schwefel(dimension),
            'michalewicz': Michalewicz(dimension),
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

class Rastrigin:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        rastrigin = 10 * self.dimension + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)
        return -rastrigin

class Schwefel:
    """Global optimum sits near x_i=420.9687, so the search domain MUST be the
    canonical [-500, 500] (enforced via CANONICAL_DOMAINS); on a small box the
    418.9829*d offset dominates and the optimum is unreachable."""
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        schwefel = 418.9829 * self.dimension - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
        return -schwefel

class Michalewicz:
    """real(x) = +sum sin(x) sin(i x^2/pi)^(2m) == -(standard Michalewicz), so
    maximising real (as the framework does) minimises the standard function --
    same orientation as the other benchmarks. Its optimum is a positive,
    dimension-dependent value (not 0); see reference_optimum(). Canonical domain
    is [0, pi] per dimension, enforced via CANONICAL_DOMAINS."""
    def __init__(self, dimension, m=10):
        self.dimension = dimension
        self.m = m

    def real(self, x):
        i = np.arange(1, self.dimension + 1)
        return np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * self.m), axis=1)

