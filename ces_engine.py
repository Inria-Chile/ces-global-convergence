import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu, eigsh
import autograd.numpy as anp
from autograd import grad

# --- Reproductibility configuration

master_seed = 42 
rng = np.random.default_rng(master_seed)


# --- CES and Replicator-Mutator equation on 1D Bounded Domain
def sol_rep_mutador(
    a, sigma, u0, c, T,
    nx=201, xmin=0.0, xmax=1.0,
    dt=0.1,
    renormalize=False
):
    c = float(c)
    Lx = xmax - xmin

    xx = np.linspace(xmin, xmax, nx)     
    dx = xx[1] - xx[0]                   

    sig = sigma(xx)

    u_0 = u0(xx)


    off = np.ones(nx - 1)
    main = -2.0 * np.ones(nx)
    A = diags([off, main, off], offsets=[-1, 0, 1], format="lil")

    A[0, 1] = 2.0
    A[-1, -2] = 2.0

    A = A.tocsr() / (dx * dx)

    I = identity(nx, format="csr")

    def trapz_on_grid(u):
        return dx * (0.5 * u[0] + np.sum(u[1:-1]) + 0.5 * u[-1])


    M = (I - (0.5 * dt * c) * A).tocsc()
    Minv = splu(M)
    P = (I + (0.5 * dt * c) * A).tocsr()

    def implicit_solve(rhs):
        return Minv.solve(rhs)

    def add_diffusion_rhs(q):
        return P @ q

    nsteps = int(np.ceil(T / dt))
    tt = np.linspace(0.0, nsteps * dt, nsteps + 1)
    U = np.zeros((nsteps + 1, nx))
    U[0] = u_0

    for n in range(nsteps):
        u = U[n]

        ubar = trapz_on_grid(sig * u)
        mass = trapz_on_grid(u)

        R = a * u * (sig - ubar)

        rhs = add_diffusion_rhs(u) + dt * R

        u_new = implicit_solve(rhs)

        if renormalize:
            u_new = u_new / trapz_on_grid(u_new)

        U[n + 1] = u_new
    return tt, xx, U
def reflection_interval(u, xmin=-2.0, xmax=2.0):
    L = xmax - xmin
    v = np.mod(u - xmin, 2.0 * L)
    return xmin + np.where(v <= L, v, 2.0 * L - v)

def sample_neumann_heat(t, x, xmin=-2.0, xmax=2.0):

    sigma = np.sqrt(2.0 * float(t))  

    u = x + sigma * rng.standard_normal(size=x.shape)
    return reflection_interval(u, xmin=xmin, xmax=xmax)



def selection_1d(x, prob):
    M = len(x) 
    idx = rng.choice(M, size=M, replace=True, p=prob)
    return x[idx], idx


def selection_mutation_step_1d(x_prev, sigma, a, c, alpha, xmin=-2.0, xmax=2.0):

    M = len(x_prev)
    a_M = a/(M**alpha)
    t_M = 1/(M**alpha)

    w_M = 1 + a_M*sigma(x_prev)

    weights = w_M / w_M.sum()

    x_sel, idx = selection_1d(x_prev, weights)

    x_next = sample_neumann_heat(c*t_M, x_sel, xmin=xmin, xmax=xmax)

    return x_next

def selection_mutation_iterator_1d(x0, sigma, a, c, alpha, N, save_times=None, xmin=-2.0, xmax=2.0):

    x = x0.copy()

    if save_times is None:
        for _ in range(N):
            x = selection_mutation_step_1d(x, sigma, a, c, alpha, xmin=xmin, xmax=xmax)
        return x

    L = len(save_times)
    markov_chain = [] 
    idx = 0 

    while idx < L and save_times[idx] == 0: 
       markov_chain.append(x.copy())
       idx += 1

    for n in range(1, N + 1):
        x = selection_mutation_step_1d(x, sigma, a, c, alpha, xmin=xmin, xmax=xmax)
        while idx < L and save_times[idx] == n:
            markov_chain.append(x.copy())
            idx += 1
        if idx == L:
           break  

    markov_chain = np.stack(markov_chain, axis=0)

    return markov_chain


def run_ces_1d(t, x0, sigma, a, c, alpha, xmin=-2.0, xmax=2.0):
    M = len(x0)
    
    is_scalar = np.isscalar(t)
    
    if is_scalar:
        N_final = int(np.floor((M**alpha) * t))
        values = selection_mutation_iterator_1d(
            x0, sigma, a, c, alpha, N_final, 
            save_times=None, xmin=xmin, xmax=xmax
        )
        return values
    else:
        N_steps = np.floor((M**alpha) * t).astype(int)
        N_max = int(np.max(N_steps))
        
        values = selection_mutation_iterator_1d(
            x0, sigma, a, c, alpha, N_max, 
            save_times=N_steps, xmin=xmin, xmax=xmax
        )
        return values

def heat_kernel_neumann(t, x, y, N=200000, xmin=-2.0, xmax=2.0):
    t = float(t)
    xmin = float(xmin)
    xmax = float(xmax)

    L = xmax - xmin

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bx, by = np.broadcast_arrays(x, y)


    N_array = np.arange(-N, N + 1, dtype=float)  

    denom = np.sqrt(4.0 * np.pi * t)

    z1 = (bx - by)[..., None] + 2.0 * L * N_array

    z2 = (bx + by - 2.0 * xmin)[..., None] + 2.0 * L * N_array

    g1 = np.exp(-(z1 * z1) / (4.0 * t)) / denom
    g2 = np.exp(-(z2 * z2) / (4.0 * t)) / denom

    return np.sum(g1 + g2, axis=-1)

def run_semi_class(a, c, sigma, nx=200, xmin=0.0, xmax=1.0, neigen=1, normalize=True):

    x = np.linspace(xmin, xmax, nx+1)
    dx = x[1] - x[0]
    sigma_x = np.asarray(sigma(x), dtype=float)

    n = nx + 1

    main = (2.0*c/dx**2) * np.ones(n)
    main[0] = main[-1] = (1.0*c/dx**2)
    off  = (-c/dx**2) * np.ones(n-1)

    L = diags([off, main, off], offsets=[-1, 0, 1], format="csc")


    L = L + diags(-a * sigma_x, 0, format="csc")


    eigen_vals, eigen_vecs = eigsh(L, k=neigen, which="SA")
    eigen_vecs = eigen_vecs.T   

    if normalize:
        w = np.ones(n); w[0] = w[-1] = 0.5
        norms = np.sqrt(dx * np.sum(w * eigen_vecs**2, axis=1, keepdims=True))
        eigen_vecs = eigen_vecs / norms

    return x, eigen_vecs, eigen_vals

#--- Error Functions

def cdf_sample(points, sample):
    x = np.sort(np.asarray(sample).ravel())
    n = x.size
    count = np.searchsorted(x, points, side="right")
    return count / n

def cdf_density(points, density):
    x = points
    dx = x[1] - x[0]

    F = np.empty_like(density, dtype=float)
    F[0] = 0.0
    F[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * dx)

    Z = F[-1]
    F /= Z
    return F

def error(pop, U, tt, xx):
    dx = xx[1] - xx[0]
    dt = tt[1]-tt[0]
    T = tt[-1]
    N = len(tt)
    w_distances = np.array([np.trapezoid(np.abs(cdf_density(xx, U[n])-cdf_sample(xx, pop[n])), xx, dx) for n in range(N)])
    error_L1  = (1/T)*np.trapezoid(w_distances, tt, dt) 
    return error_L1

# --- CES R^d

def selection_rd(x, prob):
    M = x.shape[0]
    idx = rng.choice(M, size=M, replace=True, p=prob)
    return x[idx], idx


def selection_mutation_step_rd(x_prev, sigma_fun, a, c, alpha):
    M = x_prev.shape[0]
    a_M = a / (M**alpha)
    t_M = 1.0 / (M**alpha)

    s = sigma_fun(x_prev)                
    s = np.asarray(s, dtype=float).reshape(M)

    z = -a_M * s         
    z = z - np.max(z)           
    w_M = np.exp(z)
    weights = w_M / np.sum(w_M)


    x_sel, idx = selection_rd(x_prev, weights)

    x_next = x_sel + np.sqrt(2*c*t_M) * rng.standard_normal(size=x_sel.shape)
    return x_next


def selection_mutation_iterator_rd(x0, sigma_fun, a, c, alpha, N, save_times=None):
    x = x0.copy()

    if save_times is None:
        for _ in range(N):
            x = selection_mutation_step_rd(x, sigma_fun, a, c, alpha)
        return x

    L = len(save_times)
    markov_chain = []
    idx = 0

    while idx < L and save_times[idx] == 0:
        markov_chain.append(x.copy())
        idx += 1

    for n in range(1, N + 1):
        x = selection_mutation_step_rd(x, sigma_fun, a, c, alpha)
        while idx < L and save_times[idx] == n:
            markov_chain.append(x.copy())
            idx += 1
        if idx == L:
            break

    return np.stack(markov_chain, axis=0)   


def run_ces_rd(t, x0, sigma_fun, a, c, alpha):
    t = np.asarray(t, dtype=float).ravel()
    M = x0.shape[0]
    N = np.floor((M**alpha) * t).astype(int)
    N_max = int(np.max(N))

    if t.size == 1:
        return selection_mutation_iterator_rd(x0, sigma_fun, a, c, alpha, N_max, save_times=None)

    return selection_mutation_iterator_rd(x0, sigma_fun, a, c, alpha, N_max, save_times=N)

def _eval_f(f, X):
    try:
        vals = f(X)
        vals = np.asarray(vals)
        if vals.shape == (X.shape[0],):
            return vals
    except Exception:
        pass
    return np.array([f(x) for x in X], dtype=float)


def consensus_point(X, f, beta):
    fvals = _eval_f(f, X)  
    logw = -beta * fvals
    shift = np.max(logw)
    w = np.exp(logw - shift)
    w_sum = np.sum(w)
    if not np.isfinite(w_sum):
        w = np.ones(X.shape[0], dtype=float) / X.shape[0]
    else:
        w = w / w_sum
    c = w @ X  
    return c, fvals, w


def cbo(
    f,
    X0,
    n_steps,
    dt,
    lam,
    sigma,
    beta,
    save_every=None,
):
    X = np.asarray(X0, dtype=float).copy()
    N, d = X.shape

    sqrt_dt = np.sqrt(dt)

    history = []
    def maybe_save(k, c, fvals):
        if save_every is not None and (k % save_every == 0):
            history.append(
                {
                    "step": k,
                    "X": X.copy(),
                    "c": c.copy()
                }
            )

    c, fvals, _ = consensus_point(X, f, beta)
    maybe_save(0, c, fvals)

    for k in range(1, n_steps + 1):
        c, fvals, _ = consensus_point(X, f, beta)

        diff = X - c  
        norms = np.linalg.norm(diff, axis=1)  

        W_i = rng.standard_normal(size=(N, d))
        noise = (norms[:, None]) * W_i

        X = X - lam * dt * diff + sigma * sqrt_dt * noise

        maybe_save(k, c, fvals)

    c, fvals, _ = consensus_point(X, f, beta)

    return X, c, history

def sgd_ackley(x0, lr, n_steps, a=20.0, b=0.2, c=2.0*np.pi):
    def ackley_autograd(x):
        d = x.shape[0]
        mean_sq = anp.mean(x**2)
        mean_cos = anp.mean(anp.cos(c * x))
        term1 = -a * anp.exp(-b * anp.sqrt(mean_sq))
        term2 = -anp.exp(mean_cos)
        return term1 + term2 + a + anp.e

    grad_f = grad(ackley_autograd)
    
    x = np.array(x0, dtype=float).copy()
    
    for _ in range(n_steps):
        g = grad_f(x)
        x = x - lr * g  
        
    return x