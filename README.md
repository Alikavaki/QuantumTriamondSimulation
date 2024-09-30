# Quantum Triamond Simulation
This project implements Python-based lattice quantum simulations using the Triamond lattice structure. It models a quantum lattice system, generates physical states, and computes the Hamiltonian matrix, accounting for symmetries and transformations in the system.

Note: The States and Hamiltonian classes are designed to handle one row of unit cells. For generating the Hamiltonian and states in more general cases (e.g., multiple rows or dimensions), the code will require future updates to extend its functionality.

## Classes Overview
1. Triamond
This class defines a lattice unit cell structure and handles the 3D visualization of the lattice. It includes points inside and outside the unit cell, connected by specific vectors.

Methods:
__init__(): Initializes the lattice structure by defining points and vectors.
plot(nx, ny, nz, points): Plots the 3D lattice structure for nx, ny, and nz unit cells.
2. States
The States class generates physical states for a lattice system based on Gauss's law. It checks all possible configurations of variables and validates the ones that comply with the constraints.

Methods:
__init__(num_unit): Initializes with the number of unit cells.
generate_states(): Generates valid physical states based on Gauss's law.

3. Hamiltonian
This class constructs and calculates the Hamiltonian matrix for the lattice system. It also explores translation and screw transformations to generate new quantum states and symmetries.

Methods:
__init__(num_unit, sector): Initializes the Hamiltonian with unit cells and sector.
explore_states_with_translation(): Explores all states generated by applying translation symmetry.
explore_states_with_screw_transformation(): Explores all states generated by applying screw transformations.
compute_hamiltonian(g): Computes the full Hamiltonian matrix.

## Future Updates
Currently, the States and Hamiltonian classes are only designed for one row of unit cells. The next update will extend the code to handle more general cases with multiple rows or dimensions.
