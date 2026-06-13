import sympy as sp

# variable
sigma = sp.Symbol('sigma', real=True)

# constants (C ignored since it doesn't affect extrema)
f = 27 - 18*sigma + 2*sigma**2
psi = f * sp.exp(-sigma/3)

# derivative
dpsi = sp.diff(psi, sigma)

# solve critical points
crit_points = sp.solve(sp.Eq(dpsi, 0), sigma)

print("Critical points:", crit_points)

# exact minimum candidate (choose smaller root)
sigma_min = min(crit_points)
print("Sigma at minimum:", sigma_min)

# numerical value
print("Approx value:", float(sigma_min))

# verify second derivative
d2psi = sp.diff(dpsi, sigma)
print("Second derivative at min:",
      d2psi.subs(sigma, sigma_min).evalf())