import numpy as np

# --- Fitness Functions

sigma_1 = lambda x: 0.2*np.exp(-5*x**4+x**3+2*x**2+1)
sigma_1_max = 0.74582
sigma_1_argmax = 0.52846

sigma_2 = lambda x: 0.5*np.exp(-5*(x-0.5)**2) + 0.5*np.exp(-5*(x+0.5)**2)
sigma_2_max = 0.5*(1+np.exp(-5))
sigma_2_argmax = [-0.5, 0.5]

sigma_3= lambda x: 0.8*np.exp(-10*x**2*(x-1)**4*(x+1)**4)
sigma_3_max= 0.8
sigma_3_argmax=[-1.0, 0.0, 1.0]

# --- Benchmark Function

def ackley(x, a=20.0, b=0.2, c=2.0*np.pi):
    """
    Ackley d-dimensional (forma estándar).
    f(x) = -a*exp(-b*sqrt((1/d)*sum x_i^2)) - exp((1/d)*sum cos(c*x_i)) + a + e
    Global min: x=0, f(0)=0.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    d = x.shape[-1]

    mean_sq = np.mean(x**2, axis=-1)
    mean_cos = np.mean(np.cos(c * x), axis=-1)

    term1 = -a * np.exp(-b * np.sqrt(mean_sq))
    term2 = -np.exp(mean_cos)
    return term1 + term2 + a + np.e
