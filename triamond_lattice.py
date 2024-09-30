####Importing the necessary libraries######
import numpy as np
from itertools import product, cycle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import time
from itertools import product, cycle

class Triamond:
    def __init__(self):
        #Defining the lattice unit cell
        #First:Points inside the cell
        self.w1, self.w2 = np.array([0, 0, 0]), np.array([1/2, 1/2, 1/2])
        self.b1, self.b2 = np.array([1/2, 1/4, 3/4]), np.array([0, 3/4, 1/4])
        self.r1, self.r2 = np.array([3/4, 1/2, 1/4]), np.array([1/4, 0, 3/4])
        self.g1, self.g2 = np.array([1/4, 3/4, 1/2]), np.array([3/4, 1/4, 0])
        #############################################################################################
        #We also need the points which live outside the cube
        #y---->y-1
        self.g3 = np.array([1/4, -1/4, 1/2])
        self.b3 = np.array([0, -1/4, 1/4])
        #y---> y+1
        self.r3 = np.array([1/4, 1, 3/4])
        self.w3 = np.array([0, 1, 0])
        #z--->z+1
        self.g4 = np.array([3/4, 1/4, 1])
        self.w4 = np.array([0, 0, 1])
        #z---->z-1
        self.r4 = np.array([1/4, 0, -1/4])
        self.b4 = np.array([1/2, 1/4, -1/4])
        #x---->x-1
        self.r5 = np.array([-1/4, 1/2, 1/4])
        self.g5 = np.array([-1/4, 1/4, 0])
        #x---->x+1
        self.w5 = np.array([1, 0, 0])
        self.b5 = np.array([1, 3/4, 1/4])
        ################################################################
        #Vectors
        self.b = np.array([0, -1/4, 1/4])
        self.r = np.array([1/4, 0, -1/4])
        self.g = np.array([-1/4, 1/4, 0])
        self.y = np.array([0, -1/4, -1/4])
        self.c = np.array([-1/4, 0, -1/4])
        self.m = np.array([-1/4, -1/4, 0])
        #################################################################
        #These definitions are for finding the physical states 
        self.two_unit = []
        self.trun = np.array([0, 1/2])
        #################################################################
    def plot(self, nx, ny, nz, points):
        # Only create the figure when the plot function is called
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.grid(False)
        ##############################################################
        #some parameters to plot the lattice
        # Create axis
        self.axes = [1, 1, 1]
 
        # Create Data
        self.data = np.ones(self.axes, dtype='bool')
 
        # Control Transparency
        self.alpha = 0.3
 
        # Control colour
        self.colors = np.empty(self.axes + [4], dtype=np.float32)
 
        self.colors[:] = [1, 1, 1, self.alpha]  # red
        
        for i, j, k in product(range(nx), range(ny), range(nz)):
            self.ax.quiver(self.w2[0]+i, self.w2[1]+j, self.w2[2]+k, self.b[0], self.b[1], self.b[2], color='b', arrow_length_ratio=0)#b = b1 - w2
            self.ax.quiver(self.w2[0]+i, self.w2[1]+j, self.w2[2]+k, self.r[0], self.r[1], self.r[2], color='r', arrow_length_ratio=0)#r = r1 - w2
            self.ax.quiver(self.w2[0]+i, self.w2[1]+j, self.w2[2]+k, self.g[0], self.g[1], self.g[2], color='g', arrow_length_ratio=0)#g = g1 - w2
            self.ax.quiver(self.g1[0]+i, self.g1[1]+j, self.g1[2]+k, self.c[0], self.c[1], self.c[2], color='c', arrow_length_ratio=0)#c = b2 - g1
            self.ax.quiver(self.b1[0]+i, self.b1[1]+j, self.b1[2]+k, self.m[0], self.m[1], self.m[2], color='m', arrow_length_ratio=0)#m = r2 - b1
            self.ax.quiver(self.r1[0]+i, self.r1[1]+j, self.r1[2]+k, self.y[0], self.y[1], self.y[2], color='y', arrow_length_ratio=0)#y = g2 - r1
            self.ax.quiver(self.w1[0]+i, self.w1[1]+j, self.w1[2]+k, self.b[0], self.b[1], self.b[2], color='b', arrow_length_ratio=0)#b = b1 - w2
            self.ax.quiver(self.w1[0]+i, self.w1[1]+j, self.w1[2]+k, self.r[0], self.r[1], self.r[2], color='r', arrow_length_ratio=0)#r = r1 - w2
            self.ax.quiver(self.w1[0]+i, self.w1[1]+j, self.w1[2]+k, self.g[0], self.g[1], self.g[2], color='g', arrow_length_ratio=0)#g = g1 - w2
        
            self.ax.quiver(self.r2[0]+i, self.r2[1]+j, self.r2[2]+k, self.y[0], self.y[1], self.y[2], color='y', arrow_length_ratio=0)#yellow y--->y-1
            self.ax.quiver(self.w3[0]+i, self.w3[1]+j, self.w3[2]+k, self.b[0], self.b[1], self.b[2], color='b', arrow_length_ratio=0)#blue y---->y+1
            self.ax.quiver(self.r3[0]+i, self.r3[1]+j, self.r3[2]+k, self.y[0], self.y[1], self.y[2], color='y', arrow_length_ratio=0)#yellow y---->y+1
            self.ax.quiver(self.g2[0]+i, self.g2[1]+j, self.g2[2]+k, self.c[0], self.c[1], self.c[2], color='c', arrow_length_ratio=0)#cyan z---->z-1
            self.ax.quiver(self.g4[0]+i, self.g4[1]+j, self.g4[2]+k, self.c[0], self.c[1], self.c[2], color='c', arrow_length_ratio=0)#cyan z--->z+1
            self.ax.quiver(self.w4[0]+i, self.w4[1]+j, self.w4[2]+k, self.r[0], self.r[1], self.r[2], color='r', arrow_length_ratio=0)#red z---->z+1
            self.ax.quiver(self.b2[0]+i, self.b2[1]+j, self.b2[2]+k, self.m[0], self.m[1], self.m[2], color='m', arrow_length_ratio=0)#magenta x---->x-1
            self.ax.quiver(self.w5[0]+i, self.w5[1]+j, self.w5[2]+k, self.g[0], self.g[1], self.g[2], color='g', arrow_length_ratio=0)#green x---->x+1
            self.ax.quiver(self.b5[0]+i, self.b5[1]+j, self.b5[2]+k, self.m[0], self.m[1], self.m[2], color='m', arrow_length_ratio=0)#magenta x---->x+1

            if points==True:
                self.ax.scatter(self.w1[0]+i, self.w1[1]+j, self.w1[2]+k, color='w', s=25)  # White1 point
                self.ax.scatter(self.w2[0]+i, self.w2[1]+j, self.w2[2]+k, color='w', s=25)  # White2 point
                self.ax.scatter(self.g1[0]+i, self.g1[1]+j, self.g1[2]+k, color='g', s=25)  # Green1 point
                self.ax.scatter(self.g2[0]+i, self.g2[1]+j, self.g2[2]+k, color='g', s=25)  # Green2 point
                self.ax.scatter(self.b1[0]+i, self.b1[1]+j, self.b1[2]+k, color='b', s=25)  # Blue1 point
                self.ax.scatter(self.b2[0]+i, self.b2[1]+j, self.b2[2]+k, color='b', s=25)  # Blue2 point
                self.ax.scatter(self.r1[0]+i, self.r1[1]+j, self.r1[2]+k, color='r', s=25)  # Red1 point
                self.ax.scatter(self.r2[0]+i, self.r2[1]+j, self.r2[2]+k, color='r', s=25)  # Red2 point
            
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])


        self.ax.set_xlim([-2, nx+2])
        self.ax.set_ylim([-2, ny+2])
        self.ax.set_zlim([-2, nz+2])

        self.ax.set_xlabel('x', labelpad=2)
        self.ax.set_ylabel('y', labelpad=2)
        self.ax.set_zlabel('z', labelpad=2)
        return plt.show()


###########################Defining a class for building all physical states#############################################

class States:
    def __init__(self, num_unit):
        # Initialize with the number of unit cells
        self.num_unit = num_unit
        self.valid_states = []

    def __repr__(self):
        return f"Generating physical states for {self.num_unit} unit cell(s) along one row."

    def generate_states(self):
        """
        Generates physical states for any number of unit cells along one row.
        """
        print(f"Generating physical states for {self.num_unit} unit cell(s) along one row.")
        num_variables = 6 * 2 * self.num_unit

        trun = [0, 1/2]
        for vals in product(trun, repeat=num_variables):
            valid = True
            for i in range(2 * self.num_unit):
                g_i = vals[i * 6]
                b_i = vals[i * 6 + 1]
                r_i = vals[i * 6 + 2]
                c_i = vals[i * 6 + 3]
                m_i = vals[i * 6 + 4]
                y_i = vals[i * 6 + 5]

                g_next = vals[((i + 1) * 6) % num_variables]
                m_next = vals[((i + 1) * 6 + 4) % num_variables]

                if (b_i + g_i + r_i not in (0, 1)) or \
                   (b_i + c_i + m_i not in (0, 1)) or \
                   (c_i + y_i + g_next not in (0, 1)) or \
                   (r_i + m_next + y_i not in (0, 1)):
                    valid = False
                    break

            if valid:
                self.valid_states.append(list(vals))

        return self.valid_states
################# A class for Hamiltonian ###################################################################

class Hamiltonian:
    def __init__(self, num_unit, sector):
        """
        Initializes the Hamiltonian class with the given number of unit cells and sector along x-axis.
        :param num_unit: Number of unit cells.
        :param sector: The sector requested ('vac_sector', 'ver_sector', 'hor_sector', or 'six_sector').
        """
        self.num_unit = num_unit
        self.sector = sector
        self.n_max = 2 * num_unit
        self.translation_symmetry = translation_symmetry  # Key to use translation symmetry
        # other initialization...
        self.sub_space = None
        self.magnetic_part = None
        self.electric_part = None

    def get_indices(self, cell):
        """
        Helper function to generate the indices for each variable in the unit cells.
        :param cell: Cell number.
        :return: Dictionary of indices for variables in the unit cell.
        """
        base = cell * 6  # Each unit cell starts with 6 variables
        return {
            'g': base,
            'b': base + 1,
            'r': base + 2,
            'c': base + 3,
            'm': base + 4,
            'y': base + 5
        }

    def plaq(self, color, n, s):
        """
        Generalized function for plaquettes across any number of unit cells.
        :param color: The color of the plaquette ('G', 'B', 'R', 'Y').
        :param n: The plaquette index (1 to num_unit)
        :param s: The current state list
        :return: The updated state after flipping
        """
        state = []
        n_max = 2 * self.num_unit  # Maximum number of plaquettes, double the number of unit cells

        # Get the correct interiors based on the color and plaquette index `n`
        if color == 'G':
            interiors = [
                self.get_indices(n)['b'], self.get_indices(n)['c'], self.get_indices(n)['y'], 
                self.get_indices((n + 1) % n_max)['b'], self.get_indices((n + 1) % n_max)['r'], 
                self.get_indices((n + 1) % n_max)['y'], self.get_indices((n + 1) % n_max)['c'], 
                self.get_indices(n)['r']
            ]

        elif color == 'B':
            interiors = [
                self.get_indices(n)['g'], self.get_indices((n - 1) % n_max)['y'], self.get_indices(n)['m'], self.get_indices(n)['c'], 
                self.get_indices((n + 1) % n_max)['g'], self.get_indices((n + 1) % n_max)['r'], 
                self.get_indices((n + 1) % n_max)['y'], self.get_indices((n + 1) % n_max)['c'], 
                self.get_indices((n + 1) % n_max)['m'], self.get_indices(n)['r']
            ]

        elif color == 'R':
            interiors = [
                self.get_indices(n)['b'], self.get_indices(n)['g'], self.get_indices((n - 1) % n_max)['y'], self.get_indices(n)['m'], 
                self.get_indices((n + 1) % n_max)['g'], self.get_indices((n + 1) % n_max)['b'], 
                self.get_indices((n + 1) % n_max)['m'], self.get_indices(n)['y']
            ]

        elif color == 'Y':
            interiors = [
                self.get_indices(n)['b'], self.get_indices(n)['c'], self.get_indices((n + 1) % n_max)['g'], self.get_indices((n + 1) % n_max)['r'], 
                self.get_indices((n + 2) % n_max)['m'], self.get_indices((n + 2) % n_max)['b'], 
                self.get_indices((n + 2) % n_max)['g'], self.get_indices((n + 1) % n_max)['c'], 
                self.get_indices((n + 1) % n_max)['m'], self.get_indices(n)['r']
            ]

        # Update the state based on interiors
        for i in range(len(s)):
            if i in interiors:
                state.append(self.flip(s[i]))
            else:
                state.append(s[i])

        return state

    def flip(self, val):
        """
        Flips the state value (0 <-> 1/2, 1/2 <-> 0).
        :param val: The current value.
        :return: The flipped value.
        """
        return 0 if val == 1/2 else 1/2


    def coeff(self, st, inter, exter):
        """
        Calculate the Celebs-Gordon coefficient for the transition amplitude between states.
        :param st: The current state.
        :param inter: List of interior indices.
        :param exter: List of exterior indices.
        :return: The Celebs-Gordon coefficient.
        """
        cg = 1.0
        for i in range(len(exter)):
            for j in range(len(exter[i])):
                if st[inter[i][(j + 1) % len(inter[i])]] == 0 and st[inter[i][j]] == 0:
                    cg *= np.sqrt((1 - st[exter[i][j]]) * (2 + st[exter[i][j]])) * np.sqrt(2) / 2
                elif st[inter[i][(j + 1) % len(inter[i])]] == 0 and st[inter[i][j]] == 1/2:
                    cg *= np.sqrt((1/2 + st[exter[i][j]])) ** 2
                elif st[inter[i][(j + 1) % len(inter[i])]] == 1/2 and st[inter[i][j]] == 0:
                    cg *= np.sqrt((1/2 + st[exter[i][j]])) ** 2 * 1/2
                elif st[inter[i][(j + 1) % len(inter[i])]] == 1/2 and st[inter[i][j]] == 1/2:
                    cg *= np.sqrt((2 + st[exter[i][j]]) * (1 - st[exter[i][j]])) * np.sqrt(2) / 2
                    
        return cg

    def cg_general(self, color, n, s):
        """
        Generalized function to find the Celebs-Gordon coefficient for any number of unit cells.
        :param color: The color of the plaquette ('G', 'B', 'R', 'Y').
        :param n: The plaquette index.
        :param s: The current state list.
        :return: The CG coefficient (float).
        """
        cg = 1.0
        n_max = self.n_max  # Maximum number of plaquettes
    
        # Define interiors and exteriors based on color and n
        if color == 'G' and (n % 2) == 0:
            interiors = [
                [self.get_indices(n)['b'], self.get_indices(n)['c'], self.get_indices(n)['y'], self.get_indices(n)['r']],
                [self.get_indices((n + 1) % n_max)['c'], self.get_indices((n + 1) % n_max)['b'], self.get_indices((n + 1) % n_max)['r'], self.get_indices((n + 1) % n_max)['y']]
            ]
            exteriors = [
                [self.get_indices(n)['m'], self.get_indices((n + 1) % n_max)['g'], self.get_indices((n + 1) % n_max)['m'], self.get_indices(n)['g']],
                [self.get_indices((n + 1) % n_max)['m'], self.get_indices((n + 1) % n_max)['g'], self.get_indices((n + 2) % n_max)['m'], self.get_indices((n + 2) % n_max)['g']]
            ]
        elif color == 'G' and (n % 2) == 1:
            interiors = [
                [self.get_indices(n)['c'], self.get_indices(n)['b'], self.get_indices(n)['r'], self.get_indices(n)['y']],
                [self.get_indices((n + 1) % n_max)['b'], self.get_indices((n + 1) % n_max)['c'], self.get_indices((n + 1) % n_max)['y'], self.get_indices((n + 1) % n_max)['r']]
            ]
            exteriors = [
                [self.get_indices(n)['m'], self.get_indices(n)['g'], self.get_indices((n + 1) % n_max)['m'], self.get_indices((n + 1) % n_max)['g']],
                [self.get_indices((n + 1) % n_max)['m'], self.get_indices((n + 2) % n_max)['g'], self.get_indices((n + 2) % n_max)['m'], self.get_indices((n + 1) % n_max)['g']]
            ]
        elif color == 'B':
            interiors = [
                [self.get_indices(n)['g'], self.get_indices((n - 1) % n_max)['y'], self.get_indices(n)['m'], self.get_indices(n)['c'],
                 self.get_indices((n + 1) % n_max)['g'], self.get_indices((n + 1) % n_max)['r'], self.get_indices((n + 1) % n_max)['y'], self.get_indices((n + 1) % n_max)['c'],
                 self.get_indices((n + 1) % n_max)['m'], self.get_indices(n)['r']]
            ]
            exteriors = [
                [self.get_indices((n - 1) % n_max)['c'], self.get_indices((n - 1) % n_max)['r'], self.get_indices(n)['b'], self.get_indices(n)['y'],
                 self.get_indices((n + 1) % n_max)['b'], self.get_indices((n + 2) % n_max)['m'], self.get_indices((n + 2) % n_max)['g'], self.get_indices((n + 1) % n_max)['b'],
                 self.get_indices(n)['y'], self.get_indices(n)['b']]
            ]
        elif color == 'R' and (n % 2) == 0:
            interiors = [
                [self.get_indices(n)['g'], self.get_indices((n - 1) % n_max)['y'], self.get_indices(n)['m'], self.get_indices(n)['b']],
                [self.get_indices(n)['y'], self.get_indices((n + 1) % n_max)['g'], self.get_indices((n + 1) % n_max)['b'], self.get_indices((n + 1) % n_max)['m']]
            ]
            exteriors = [
                [self.get_indices((n - 1) % n_max)['c'], self.get_indices((n - 1) % n_max)['r'], self.get_indices(n)['c'], self.get_indices(n)['r']],
                [self.get_indices(n)['c'], self.get_indices((n + 1) % n_max)['r'], self.get_indices((n + 1) % n_max)['c'], self.get_indices(n)['r']]
            ]
        elif color == 'R' and (n % 2) == 1:
            interiors = [
                [self.get_indices((n - 1) % n_max)['y'], self.get_indices(n)['g'], self.get_indices(n)['b'], self.get_indices(n)['m']],
                [self.get_indices((n + 1) % n_max)['g'], self.get_indices(n)['y'], self.get_indices((n + 1) % n_max)['m'], self.get_indices((n + 1) % n_max)['b']]
            ]
            exteriors = [
                [self.get_indices((n - 1) % n_max)['c'], self.get_indices(n)['r'], self.get_indices(n)['c'], self.get_indices((n - 1) % n_max)['r']],
                [self.get_indices(n)['c'], self.get_indices(n)['r'], self.get_indices((n + 1) % n_max)['c'], self.get_indices((n + 1) % n_max)['r']]
            ]
        elif color == 'Y':
            interiors = [
                [self.get_indices(n)['b'], self.get_indices(n)['c'], self.get_indices((n + 1) % n_max)['g'], self.get_indices((n + 1) % n_max)['r'],
                 self.get_indices((n + 2) % n_max)['m'], self.get_indices((n + 2) % n_max)['b'], self.get_indices((n + 2) % n_max)['g'], self.get_indices((n + 1) % n_max)['c'],
                 self.get_indices((n + 1) % n_max)['m'], self.get_indices(n)['r']]
            ]
            exteriors = [
                [self.get_indices(n)['m'], self.get_indices(n)['y'], self.get_indices((n + 1) % n_max)['b'], self.get_indices((n + 1) % n_max)['y'],
                 self.get_indices((n + 2) % n_max)['c'], self.get_indices((n + 2) % n_max)['r'], self.get_indices((n + 1) % n_max)['y'], self.get_indices((n + 1) % n_max)['b'],
                 self.get_indices(n)['y'], self.get_indices(n)['g']]
            ]
    
        # Apply Celebs-Gordon coefficients using the `coeff` function
        cg *= self.coeff(s, interiors, exteriors)
    
        return cg




    def initialize_states(self):
        """
        Initialize four states: q0, qv, qh, and q6 with specific non-zero components (1/2) depending on the number of unit cells.
        :return: The requested state based on the sector.
        """
        num_variables = 12 * self.num_unit  # 12 variables per unit cell

        # Initialize states
        q0 = [0] * num_variables
        qv = [0] * num_variables
        qv[1] = qv[2] = qv[3] = qv[5] = 1/2
        qh = [0] * num_variables
        qh[0] = qh[1] = qh[4] = qh[-1] = 1/2
        q6 = [0] * num_variables
        q6[0] = q6[2] = q6[3] = q6[4] = q6[5] = q6[-1] = 1/2

        if self.sector == 'vac_sector':
            return q0
        elif self.sector == 'ver_sector':
            return qv
        elif self.sector == 'hor_sector':
            return qh
        elif self.sector == 'six_sector':
            return q6
        else:
            raise ValueError("Invalid sector. Please choose from 'vac_sector', 'ver_sector', 'hor_sector', or 'six_sector'.")

    def plaq_opp_unit(self, initial_state):
        """
        Applies all operators to the initial state and generates new states iteratively until no new states are found.
        :param initial_state: The starting state.
        :return: A list of all unique states.
        """
        color_set = ['G', 'B', 'R', 'Y']
        states = [initial_state]
        state_queue = [initial_state]

        while state_queue:
            current_state = state_queue.pop(0)

            for color in color_set:
                for n in range(self.n_max):
                    new_state = self.plaq(color, n, current_state)  # Use self.plaq here
                    
                    if new_state not in states:
                        states.append(new_state)
                        state_queue.append(new_state)

        return states

    def cg_finder(self, s, block):
        """
        Finds the Clebsch-Gordan coefficients for the given state.
        :param s: Current state.
        :param block: The block of states.
        :return: A sorted list of tuples containing indices and CG coefficients.
        """
        ite, cof = [], []
        x = range(len(block))
        xy = list(zip(x, block))

        color_set = ['G', 'B', 'R', 'Y']
    
        for c, n in product(color_set, range(self.n_max)):
            sp = self.plaq(c, n, s)  # Use self.plaq here
            for i, j in xy:
                if sp == j:
                    ite.append(i)
                    cof.append(self.cg_general(c, n, s))  # Use self.cg_general here
        
        xyz = list(zip(ite, cof))
        xyz.sort()         
                    
        return xyz

    def magnetic(self, block):
        """
        Generalized function to calculate the magnetic component for any number of unit cells.
        :param block: The sub-space of states.
        :return: Magnetic matrix as a NumPy array.
        """
        mag = []
        row_size = len(block)

        for i in range(len(block)):
            row = np.zeros(row_size)
            
            for m, o in self.cg_finder(block[i], block):
                row[m] = o
                
            mag.append(row)

        return np.array(mag)


    def linear_translation(self, lst):
        """
        Applies a cyclic linear translation to the list, shifting each chunk (g, b, r, c, m, y) to the next one,
        and the last chunk moves to the first position.
        """
        chunk_size = 6  # (g, b, r, c, m, y) as one chunk
        num_chunks = len(lst) // chunk_size
        
        # Create a copy of the list to store the shifted result
        shifted_lst = lst[:]  # Make a copy to avoid modifying the original list
        
        # Loop over each chunk, shifting its elements to the next chunk's position
        for i in range(num_chunks - 1):
            # Calculate the starting indices for the current chunk and the next chunk
            src_start = i * chunk_size
            dest_start = (i + 1) * chunk_size
            
            # Move the current chunk to the next position
            shifted_lst[dest_start:dest_start + chunk_size] = lst[src_start:src_start + chunk_size]
        
        # Move the last chunk to the first position
        shifted_lst[0:chunk_size] = lst[-chunk_size:]
        
        return shifted_lst

    def screw_transformation(self, lst):
        """
        Applies the screw transformation to the list, shifting each element according to the screw transformation, Which is pi/2 rotation around x-axis and moving the entire lattice            along that axis one-fourth of a unit-cell.
        """
        chunk_size = 6  # (g, b, r, c, m, y) as one chunk
        num_chunks = len(lst) // chunk_size

        if len(lst) % chunk_size != 0:
            raise ValueError("List length must be divisible by 6")

        # Split the list into chunks
        chunks = [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        
        # Create a new list to store the transformed result
        transformed_lst = [None] * len(lst)

        for i in range(num_chunks):
            current_chunk = chunks[i]
            next_chunk = chunks[(i + 1) % num_chunks]  # Wrap around for the last chunk

            # Perform screw transformation
            transformed_lst[i * chunk_size + 2] = current_chunk[0]  # 1st -> 3rd in the same chunk
            transformed_lst[i * chunk_size + 5] = current_chunk[1]  # 2nd -> 6th in the same chunk
            transformed_lst[(i + 1) % num_chunks * chunk_size + 4] = current_chunk[2]  # 3rd -> 5th in the next chunk
            transformed_lst[(i + 1) % num_chunks * chunk_size + 0] = current_chunk[3]  # 4th -> 1st in the next chunk
            transformed_lst[i * chunk_size + 3] = current_chunk[4]  # 5th -> 4th in the same chunk
            transformed_lst[(i + 1) % num_chunks * chunk_size + 1] = current_chunk[5]  # 6th -> 2nd in the next chunk

        return transformed_lst

    def explore_states_with_translation(self):
        initial_state = self.initialize_states()
        all_states = self.plaq_opp_unit(initial_state)
        
        # Set to track processed states to prevent duplicates
        processed_states = set()
        all_symmetry_indices = []

        for ind, s in enumerate(all_states):
            if tuple(s) in processed_states:
                continue  # Skip if this state has already been processed

            current_state = s
            indices = []

            # Find all states generated by translations
            while tuple(current_state) not in processed_states:
                processed_states.add(tuple(current_state))
                indices.append(all_states.index(current_state))  # Store the index of the current state

                # Apply translation
                current_state = self.linear_translation(current_state)

            # Store the indices group for this symmetry set
            if indices:
                all_symmetry_indices.append(indices)

        return all_symmetry_indices

    def explore_states_with_screw_transformation(self):
        initial_state = self.initialize_states()
        all_states = self.plaq_opp_unit(initial_state)

        # Set to track processed states to prevent duplicates
        processed_states = set()
        all_symmetry_indices = []

        for ind, s in enumerate(all_states):
            if tuple(s) in processed_states:
                continue  # Skip if this state has already been processed

            current_state = s
            indices = []

            # Find all states generated by screw transformations
            while tuple(current_state) not in processed_states:
                processed_states.add(tuple(current_state))
                indices.append(all_states.index(current_state))  # Store the index of the current state

                # Apply screw transformation
                current_state = self.screw_transformation(current_state)

            # Store the indices group for this symmetry set
            if indices:
                all_symmetry_indices.append(indices)

        return all_symmetry_indices

    def compute_hamiltonian(self, g):
        """
        Computes the Hamiltonian matrix (electric - (1/g**4) *magnetic).
        :return: The Hamiltonian matrix.
        """
        # Initialize the states
        ini_state = self.initialize_states()
        
        # Get the sub_space
        self.sub_space = self.plaq_opp_unit(ini_state)
        
        # Calculate the magnetic part
        self.magnetic_part = self.magnetic(self.sub_space)

        # Calculate the electric part
        h_e = [sum(1 for val in state if val == 1/2) for state in self.sub_space]
        self.electric_part = np.diag(h_e)

        # Return the total Hamiltonian
        return self.electric_part - 1/g**4 * self.magnetic_part
            






















