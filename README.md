# Lindblad equations for TlF

This code is intended to be used to follow the density-matrix evolution of an
ensemble of TlF molecules as they experience time-varying electromagnetic
fields. The simulation is broken down into several numbered Jupyter notebooks:

1. Calculate the Hamiltonians encoding the internal structure of the molecule,
   as well as its interactions with the external fields. This only needs to be
   done once, and the subsequent steps use the pre-calculated matrices.

   *Inputs:* the upper limit of $J$, entered at the top of the notebook.

   *Outputs:* the Hamiltonian matrices for the X and B states, stored as binary
   NumPy (`.npy`) files in the `data/1_Hamiltonians` directory.

2. Write the explicit expression the master equation in Lindblad form (a.k.a, by
   synecdoche, the "Bloch equations").

   *Inputs:* the range of states to be included in the simulation, and the
   form of the time-dependent external fields.

   *Outputs:* a set of coupled differential equations, written in Julia, and
   stored in `data/2_equations`.

3. Solve the differential equations, and plot/postprocess the results.

   *Inputs:* the name of the Julia file and ODE solver parameters, entered in
   the beginning of the notebook.

   *Outputs:* stored in `data/3_solutions` and `data/4_plots`.
