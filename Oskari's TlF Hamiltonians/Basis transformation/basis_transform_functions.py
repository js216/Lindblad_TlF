"""
Contains functions for transforming Hamiltonian between different bases
"""

import sys
import numpy as np
sys.path.append('../molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from tqdm.notebook import tqdm

def generate_transform_matrix(basis1, basis2):
    """
    Function that generates a transform matrix that takes Hamiltonian expressed
    in basis1 to basis2: H_2 = S.conj().T @ H_1 @ S

    inputs:
    basis1 = list of basis states that defines basis1
    basis2 = list of basis states that defines basis2

    returns:
    S = transformation matrix that takes Hamiltonian (or any operator) from basis1 to basis2
    """

    #Check that the two bases have the same dimension
    if len(basis1) != len(basis2):
        print("Bases don't have the same dimension")
        return 1
    
    #Initialize S
    S = np.zeros((len(basis1), len(basis1)))

    #Loop over the bases and calculate inner products
    for i, state1 in enumerate(tqdm(basis1)):
        for j, state2 in enumerate(basis2):
            S[i,j] = state1 @ state2

    #Return the transform matrix
    return S

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

