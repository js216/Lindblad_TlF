"""
Functions for integrating Optical Bloch Equations using the method outlined in
John Barry's thesis, section 3.4
"""

import sys
import numpy as np
sys.path.append('./molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from functions import find_state_idx_from_state, vector_to_state
from matrix_element_functions import ED_ME_coupled
from tqdm.notebook import tqdm
from scipy import constants
from numpy import sqrt, exp
# from sympy import exp, sqrt
from scipy.sparse import kron, eye, coo_matrix, csr_matrix
# from numpy import kron, eye
from scipy.special import jv

def reduced_basis_hamiltonian(basis_ori, H_ori, basis_red):
    """
    Function that outputs Hamiltonian for a sub-basis of the original basis

    inputs:
    basis_ori = original basis (list of states)
    H_ori = Hamiltonian in original basis
    basis_red = sub-basis of original basis (list of states)

    outputs:
    H_red = Hamiltonian in sub-basis
    """

    #Determine the indices of each of the reduced basis states
    index_red = np.zeros(len(basis_red), dtype = int)
    for i, state_red in enumerate(basis_red):
        index_red[i] = basis_ori.index(state_red)

    #Initialize matrix for Hamiltonian in reduced basis
    H_red = np.zeros((len(basis_red),len(basis_red)), dtype = complex)

    #Loop over reduced basis states and pick out the correct matrix elements
    #for the Hamiltonian in the reduced basis
    for i, state_i in enumerate(basis_red):
        for j, state_j in enumerate(basis_red):
            H_red[i,j] = H_ori[index_red[i], index_red[j]]

    return H_red

def optical_coupling_matrix(QN, ground_states, excited_states, pol_vec = np.array([0,0,1]), reduced = False):
    """
    Function that generates the optical coupling matrix for given ground and excited states

    inputs:
    QN = list of states that defines the basis for the calculation
    ground_states = list of ground states that are coupled to the excited states (i.e. laser is resonant)
    excited_states = list of excited states that are coupled to the ground states

    outputs:
    H = coupling matrix
    """

    #Initialize the coupling matrix
    H = np.zeros((len(QN),len(QN)), dtype = complex)

    #Start looping over ground and excited states
    for ground_state in ground_states:
        i = QN.index(ground_state)
        ground_state = ground_state.remove_small_components(tol = 1e-5)
        for excited_state in excited_states:
            j = QN.index(excited_state)
            excited_state = excited_state.remove_small_components(tol = 1e-5)

            #Calculate matrix element and add it to the Hamiltonian
            H[i,j] = ED_ME_mixed_state(ground_state, excited_state, pol_vec = pol_vec, reduced = reduced)

    #Make H hermitian
    H = H + H.conj().T

    return H
                
def ED_ME_mixed_state(bra, ket, pol_vec = np.array([1,1,1]), reduced = False):
    """
    Calculates electric dipole matrix elements between mixed states

    inputs:
    bra = state object
    ket = state object
    pol_vec = polarization vector for the light that is driving the transition (the default is useful when calculating branching ratios)

    outputs:
    ME = matrix element between the two states
    """
    ME = 0
    bra = bra.transform_to_omega_basis()
    ket = ket.transform_to_omega_basis()
    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += amp_bra.conjugate()*amp_ket*ED_ME_coupled(basis_bra, basis_ket, pol_vec = pol_vec, rme_only = reduced)

    return ME

def calculate_BR(excited_state, ground_states, tol = 1e-5):
    """
    Function that calculates branching ratios from the given excited state to the given ground states

    inputs:
    excited_state = state object representing the excited state that is spontaneously decaying
    ground_states = list of state objects that should span all the states to which the excited state can decay

    returns:
    BRs = list of branching ratios to each of the ground states
    """

    #Initialize container for matrix elements between excited state and ground states
    MEs = np.zeros(len(ground_states), dtype = complex)

    #loop over ground states
    for i, ground_state in enumerate(ground_states):
        MEs[i] = ED_ME_mixed_state(ground_state.remove_small_components(tol = tol),excited_state.remove_small_components(tol = tol))
    
    #Calculate branching ratios
    BRs = np.abs(MEs)**2/(np.sum(np.abs(MEs)**2))

    return BRs


def calculate_decay_rate(excited_state, ground_states):
    """
    Function to calculate decay rates so can check that all excited states have the same decay rates
    """

    #Initialize container fo matrix elements between excited state and ground states
    MEs = np.zeros(len(ground_states), dtype = complex)

    #loop over ground states
    for i, ground_state in enumerate(ground_states):
        MEs[i] = ED_ME_mixed_state(ground_state,excited_state)

    rate = np.sum(np.abs(MEs)**2)
    return rate


def find_exact_states(states_approx, H, QN, V_ref = None):
    """
    Function for finding the closest eigenstates corresponding to states_approx

    inputs:
    states_approx = list of State objects
    H = Hamiltonian whose eigenstates are used (should be diagonal in basis QN)
    QN = List of State objects that define basis for H
    V_ref = matrix that defines what the matrix representing Qn should look like (used for keeping track of ordering)

    returns:
    states = eigenstates of H that are closest to states_aprox in a list
    """
    states = []
    for state_approx in states_approx:
        i = find_state_idx_from_state(H, state_approx, QN, V_ref = V_ref)
        states.append(QN[i])

    return states

def find_exact_states_V(states_approx, H, V, QN, V_ref = None):
    """
    Function for finding the closest eigenstates corresponding to states_approx

    inputs:
    states_approx = list of State objects
    H = Hamiltonian whose eigenstates are used (should be diagonal in basis QN)
    QN = List of State objects that define basis for H
    V_ref = matrix that defines what the matrix representing Qn should look like (used for keeping track of ordering)

    returns:
    states = eigenstates of H that are closest to states_aprox in a list
    """
    states = []
    for state_approx in states_approx:
        i = find_state_idx_from_state(H, state_approx, QN, V_ref = V_ref)
        V = V[:,i]
        # vector_to_state(V,QN).print_state()
        states.append(vector_to_state(V,QN))

    return states


