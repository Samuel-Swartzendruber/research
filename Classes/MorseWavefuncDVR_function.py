#%%
import numpy as np

# Physical constants (SI)
HBAR = 1.055e-34      # J*s
ANGSTROM_TO_M = 1.0e-10
EV_TO_J = 1.602e-19   # J/eV

def morse_potential(R_A, De_eV, a_Ainv, Re_A):
    """
    Morse potential V(R) in Joules.

    Parameters
    ----------
    R_A    : array_like
        Internuclear distance grid in Angstrom.
    De_eV  : float
        Dissociation energy D_e in eV.
    a_Ainv : float
        Morse range parameter a in Angstrom^-1.
    Re_A   : float
        Equilibrium bond length R_e in Angstrom.
    """
    De_J = De_eV * EV_TO_J
    return De_J * (1.0 - np.exp(-a_Ainv * (R_A - Re_A)))**2

def morse_hamiltonian(R_A, mu, De_eV, a_Ainv, Re_A):
    """
    Build Morse Hamiltonian H (in Joules) on a 1D grid R_A (Angstrom).

    Parameters
    ----------
    R_A    : array_like
        Internuclear distance grid in Angstrom (uniform spacing).
    mu     : float
        Reduced mass in kg.
    De_eV  : float
        Dissociation energy in eV.
    a_Ainv : float
        Morse parameter a in Angstrom^-1.
    Re_A   : float
        Equilibrium distance in Angstrom.

    Returns
    -------
    H  : (N,N) ndarray
        Hamiltonian matrix in Joules.
    V  : (N,) ndarray
        Potential array in Joules.
    """
    R_A = np.asarray(R_A)
    N = R_A.size
    dR_A = R_A[1] - R_A[0]
    dR = dR_A * ANGSTROM_TO_M  # convert to meters

    # Kinetic energy operator via 2nd-order finite difference
    coef = - (HBAR**2) / (2.0 * mu * dR**2)
    T = np.zeros((N, N), dtype=float)
    np.fill_diagonal(T, -2.0)
    np.fill_diagonal(T[1:], 1.0)
    np.fill_diagonal(T[:,1:], 1.0)
    T *= coef

    # Potential
    V = morse_potential(R_A, De_eV, a_Ainv, Re_A)
    H = T + np.diag(V)

    return H, V

def morse_wavefunction(R_A, n, mu, De_eV, a_Ainv, Re_A):
    """
    Return normalized nth Morse eigenfunction on a given R grid.

    Parameters
    ----------
    R_A    : array_like
        Internuclear distance grid in Angstrom (uniform spacing).
    n      : int
        Vibrational quantum number (0 = ground state).
    mu     : float
        Reduced mass in kg.
    De_eV  : float
        Dissociation energy in eV.
    a_Ainv : float
        Morse parameter a in Angstrom^-1.
    Re_A   : float
        Equilibrium bond length in Angstrom.

    Returns
    -------
    psi_n  : (N,) ndarray
        Normalized nth eigenfunction psi_n(R_A).
    E_n    : float
        Eigenvalue (energy) of state n in Joules.
    """
    R_A = np.asarray(R_A)
    H, V = morse_hamiltonian(R_A, mu, De_eV, a_Ainv, Re_A)

    # Diagonalize
    E, psi = np.linalg.eigh(H)
    # Sort
    idx = np.argsort(E)
    E_sorted = E[idx]
    psi_sorted = psi[:, idx]

    # Extract nth state
    psi_n = psi_sorted[:, n]

    # Normalize on R_A (Angstrom grid)
    norm = np.sqrt(np.trapz(np.abs(psi_n)**2, x=R_A))
    psi_n /= norm

    return psi_n, E_sorted[n]