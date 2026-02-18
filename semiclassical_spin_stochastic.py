from mpi4py import MPI
import symengine
from jitcsde import y, jitcsde
import numpy as np
import os
import matplotlib.pyplot as plt

# Generates an anti-symmetric adjacency matrix of a graph with ring topology
def anti_symmetric_ring_matrix(N):
    """
    Generates the adjacency matrix of a ring with each node with positive links pointing in the clockwise direction and
    negative links pointing in the counter-clockwise direction.

    :param N: Number of nodes
    :return: Adjacency matrix of an anti-symmentric ring
    """
    graph = np.zeros((N, N))
    graph[0, -1] = 1
    graph[-1, 0] = -1
    for i in range(N - 1):
        j = i + 1
        graph[i, j] = -1.0
        graph[j, i] = 1.0
    return graph

def decompose_coupling_matrix(coupling_matrix):
    """
    Generates the matrices C^a, C^h+, and C^h- from coupling matrix C as defined in section S1.

    :param coupling_matrix: Coupling matrix C that will be decomposed.
    :return: The matrices C^a, C^h+, and C^h-.
    """
    coupling_h = 0.5*(coupling_matrix + np.conjugate(coupling_matrix.T))
    coupling_a = 0.5*(coupling_matrix - np.conjugate(coupling_matrix.T))
    vals, vecs = np.linalg.eigh(coupling_h)
    coupling_hp = np.zeros((len(vals), len(vals)), dtype=np.complex128)
    coupling_hm = np.zeros((len(vals), len(vals)), dtype=np.complex128)
    for i in range(len(vals)):
        if vals[i]>0:
            coupling_hp += vals[i] * np.kron(np.reshape(vecs[:, i], (num_spins, 1)), np.conjugate(vecs[:, i]))
        elif vals[i]<0:
            coupling_hm -= vals[i] * np.kron(np.reshape(vecs[:, i], (num_spins, 1)), np.conjugate(vecs[:, i]))
    return coupling_hp, coupling_hm, coupling_a

def mu_x_classical(y, curr_spin, num_spins, omega, gamma_g, gamma_d, coupling_matrix, epsilon):
    """
    This returns the symbolic expression for the real component Eq. 8b in the limit of S -> infinity.

    :param y: 2*num_spin dimensional vector which contains the x and y components of classical each spin.
    :param curr_spin: An index whose value is between 0 and num_spins-1
    :param num_spins: Total number of spins
    :param omega: Angular frequency of the current spin
    :param gamma_g: Linear gain rate of current spin
    :param gamma_d: Non-linear damping rate of current spin.
    :param coupling_matrix: Coupling matrix C of total system
    :param epsilon: Regularization parameter which is >=0. Epsilon equals 0 corresponds to no regularization.
    :return: Symbolic expression for the curr_spin x-component used in the jitcsde integration module
    """
    mod_y_square = (y(curr_spin)*y(curr_spin) + y(curr_spin+num_spins)*y(curr_spin+num_spins))
    f_x = y(curr_spin)*(gamma_g - 4*gamma_d*mod_y_square/((1 + mod_y_square)*(1 + mod_y_square))) + y(curr_spin+num_spins)*omega - (2.718281828459045**(epsilon*mod_y_square) - 1)*y(curr_spin)/symengine.sqrt(mod_y_square)
    for i in range(num_spins):
        new_term = (np.real(coupling_matrix[curr_spin, i])*(y(i) + y(curr_spin)*y(curr_spin)*y(i) - y(curr_spin+num_spins)*y(curr_spin+num_spins)*y(i) + 2*y(curr_spin)*y(curr_spin+num_spins)*y(i+num_spins)) + np.imag(coupling_matrix[curr_spin, i])*(-y(i+num_spins)+ y(curr_spin+num_spins)*y(curr_spin+num_spins)*y(i+num_spins) - y(curr_spin)*y(curr_spin)*y(i+num_spins) + 2*y(curr_spin)*y(curr_spin+num_spins)*y(i)))
        new_term /= 1 + y(i)*y(i) + y(i+num_spins)*y(i+num_spins)
        f_x += new_term
    return f_x

def mu_y_classical(y, curr_spin, num_spins, omega, gamma_g, gamma_d, coupling_matrix, epsilon):
    """
    This returns the symbolic expression for the imaginary component of Eq. 8b in the limit of S -> infinity.

    :param y: 2*num_spin dimensional vector which contains the x and y components of classical each spin.
    :param curr_spin: An index whose value is between 0 and num_spins-1
    :param num_spins: Total number of spins
    :param omega: Angular frequency of the current spin
    :param gamma_g: Linear gain rate of current spin
    :param gamma_d: Non-linear damping rate of current spin.
    :param coupling_matrix: Coupling matrix C of total system
    :param epsilon: Regularization parameter which is >=0. Epsilon equals 0 corresponds to no regularization.
    :return: Symbolic expression for the curr_spin y-component used in the jitcsde integration module
    """

    mod_y_square = (y(curr_spin) * y(curr_spin) + y(curr_spin + num_spins) * y(curr_spin + num_spins))
    f_y = y(curr_spin+num_spins)*(gamma_g - 4*gamma_d*mod_y_square/((1 + mod_y_square)*(1 + mod_y_square))) - y(curr_spin)*omega - (2.718281828459045**(epsilon*mod_y_square) - 1)*y(curr_spin+num_spins)/symengine.sqrt(mod_y_square)
    for i in range(num_spins):
        new_term = (np.real(coupling_matrix[curr_spin, i])*(y(i+num_spins) + y(curr_spin+num_spins)*y(curr_spin+num_spins)*y(i+num_spins) - y(curr_spin)*y(curr_spin)*y(i+curr_spin) + 2*y(curr_spin)*y(curr_spin+num_spins)*y(i+num_spins)) + np.imag(coupling_matrix[curr_spin, i])*(y(i) - y(curr_spin)*y(curr_spin)*y(i) + y(curr_spin+num_spins)*y(curr_spin+num_spins)*y(i) - 2*y(curr_spin)*y(curr_spin+num_spins)*y(i+num_spins)))
        new_term /= 1 + y(i)*y(i) + y(i+num_spins)*y(i+num_spins)
        f_y += new_term
    return f_y

def mu_x(y, curr_spin, num_spins, omega, gamma_g_mod, gamma_d, spin, coupling_matrix, epsilon):
    """
    This returns the symbolic expression for the real component Eq. 8b for arbitrary S.

    :param y: 2*num_spin dimensional vector which contains the x and y components of classical each spin.
    :param curr_spin: An index whose value is between 0 and num_spins-1
    :param num_spins: Total number of spins
    :param omega: Angular frequency of the current spin
    :param gamma_g: Linear gain rate of current spin
    :param gamma_d: Non-linear damping rate of current spin.
    :param spin: Value of spin at which to calculate mu_x.
    :param coupling_matrix: Coupling matrix C of total system
    :param epsilon: Regularization parameter which is >=0. Epsilon equals 0 corresponds to no regularization.
    :return: Symbolic expression for the curr_spin x-component used in the jitcsde integration module
    """
    mod_y_square = (y(curr_spin)*y(curr_spin) + y(curr_spin+num_spins)*y(curr_spin+num_spins))
    f_x = y(curr_spin)*(gamma_g_mod - gamma_d*((2*spin-1)/spin)*mod_y_square*((mod_y_square + 2*spin)/((1 + mod_y_square)*(1 + mod_y_square)))/spin) + y(curr_spin+num_spins)*omega - (2.718281828459045**(epsilon*(y(curr_spin)*y(curr_spin) + y(curr_spin+num_spins)*y(curr_spin+num_spins))) - 1)*y(curr_spin)/symengine.sqrt(y(curr_spin)*y(curr_spin) + y(curr_spin+num_spins)*y(curr_spin+num_spins))
    for i in range(num_spins):
        new_term = (np.real(coupling_matrix[curr_spin, i])*(y(i) + y(curr_spin)*y(curr_spin)*y(i) - y(curr_spin+num_spins)*y(curr_spin+num_spins)*y(i) + 2*y(curr_spin)*y(curr_spin+num_spins)*y(i+num_spins)) + np.imag(coupling_matrix[curr_spin, i])*(-y(i+num_spins)+ y(curr_spin+num_spins)*y(curr_spin+num_spins)*y(i+num_spins) - y(curr_spin)*y(curr_spin)*y(i+num_spins) + 2*y(curr_spin)*y(curr_spin+num_spins)*y(i)))
        new_term /= 1 + y(i)*y(i) + y(i+num_spins)*y(i+num_spins)
        f_x += new_term
    return f_x

def mu_y(y, curr_spin, num_spins, omega, gamma_g_mod, gamma_d, spin, coupling_matrix, epsilon):
    """
    This returns the symbolic expression for the imaginary component Eq. 8b for arbitrary S.

    :param y: 2*num_spin dimensional vector which contains the x and y components of classical each spin.
    :param curr_spin: An index whose value is between 0 and num_spins-1
    :param num_spins: Total number of spins
    :param omega: Angular frequency of the current spin
    :param gamma_g: Linear gain rate of current spin
    :param gamma_d: Non-linear damping rate of current spin.
    :param spin: Value of spin at which to calculate mu_y.
    :param coupling_matrix: Coupling matrix C of total system
    :param epsilon: Regularization parameter which is >=0. Epsilon equals 0 corresponds to no regularization.
    :return: Symbolic expression for the curr_spin y-component used in the jitcsde integration module
    """

    mod_y_square = (y(curr_spin) * y(curr_spin) + y(curr_spin + num_spins) * y(curr_spin + num_spins))
    f_y = y(curr_spin+num_spins)*(gamma_g_mod - gamma_d*((2*spin-1)/spin)*mod_y_square*((mod_y_square + 2*spin)/((1 + mod_y_square)*(1 + mod_y_square)))/spin) - y(curr_spin)*omega - (2.718281828459045**(epsilon*(y(curr_spin)*y(curr_spin) + y(curr_spin+num_spins)*y(curr_spin+num_spins))) - 1)*y(curr_spin+num_spins)/symengine.sqrt(y(curr_spin)*y(curr_spin) + y(curr_spin+num_spins)*y(curr_spin+num_spins))
    for i in range(num_spins):
        new_term = (np.real(coupling_matrix[curr_spin, i])*(y(i+num_spins) + y(curr_spin+num_spins)*y(curr_spin+num_spins)*y(i+num_spins) - y(curr_spin)*y(curr_spin)*y(i+curr_spin) + 2*y(curr_spin)*y(curr_spin+num_spins)*y(i+num_spins)) + np.imag(coupling_matrix[curr_spin, i])*(y(i) - y(curr_spin)*y(curr_spin)*y(i) + y(curr_spin+num_spins)*y(curr_spin+num_spins)*y(i) - 2*y(curr_spin)*y(curr_spin+num_spins)*y(i+num_spins)))
        new_term /= 1 + y(i)*y(i) + y(i+num_spins)*y(i+num_spins)
        f_y += new_term
    return f_y

def mat_diff(x, y, gamma_g, gamma_d, s):
    """
    Returns the entry of the noise matrix assicated with Eq. 8c in the main text.

    :param x: x-component of the current spin, i.e. x == y(curr_spin)
    :param y: y-component of the current spin, i.e. x == y(curr_spin + num_spins)
    :param gamma_g: Gain rate of the current spin
    :param gamma_d: Danping rate of the current spin
    :param s: Spin quantum number
    :return: Symbolic expression for the real or imaginary component of Eq. 8c.
    """
    mod_gamma_square = x*x + y*y
    func = (4*s*(s-1) - 2*s*mod_gamma_square) / ((1 + mod_gamma_square)*(1 + mod_gamma_square))
    element1 = 0.5*(gamma_g + gamma_d*mod_gamma_square*mod_gamma_square*mod_gamma_square*(1/s**2)*(func + 1))/spin
    return element1

if __name__ == "__main__":
    # Set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set up parameters for job
    total_num_traj = 400 # Number of trajectories with random initial conditions to average over
    num_spins = 3 # Number of spins. This is 3 throughout the main text and supplement.
    spin = 20 # Spin quantum number. Higher spin will have better agreement with the quantum model in Eq. 4 in the main text
    omega = 1 # Natural freqency of each oscillator
    gamma_d = 2 # Non-linear damping rate of each oscillator
    gamma_g = 1 # Linear gain rate of each oscillator
    coupling_strength = 1 # Overall pre-factor with which to multiply the coupling matrix.
    delta_t = 0.01 # The size of the time-step between which samples are taken from the trajectory.
    t_final = 800 # Final time at which a single trajectory simulation will cease.
    times = np.arange(0, t_final, delta_t) # List of times at which samples will be taken
    cut_time = 20 # A cut-off which excludes all data taken before this time
    cut_ind = int(cut_time/delta_t)
    epsilon = 0.001 # Regularization. epsilon = 0 corresponds to no regularization
    p = 1 # An overall prefactor which multiplies the noise terms. p = 1 for all results in the paper and supplement
    fluctuation_tolerance = 1e-5 # Error threshold on the calculated r values. If r goes below the fluctuation_tolerance,
                                 # then the calculation at the current data point will cease.
    r_vals_output = "grid_vals_{}.out".format(spin) # Name of the output file to which values of r_vals_output will be printed

    calculate_pairwise_vals = True # Option to calculate the additional pairwise vals in Fig.

    delta_max = 2 * coupling_strength
    delta_min = -2 * coupling_strength
    grid_step = 0.01 * (delta_max - delta_min) #0.05 * (delta_max - delta_min)
    num_points = int(np.floor((delta_max - delta_min) / grid_step))
    delta_one_arr = np.linspace(delta_min, delta_max, num_points)
    delta_two_arr = np.linspace(delta_min, delta_max, num_points)
    r_vals = None
    if rank==0:
        r_vals = np.zeros((len(delta_one_arr), len(delta_two_arr)))

############################################################################################
    # Code for loading in points from previous job. This entire chunk has rank 0 check to see if there is an r_vals_output
    # file from a previous incomplete job and then read those values in if it exists. The remaining unfinished points are
    # then distributed equally among the processes.
    job_coords = None
    if rank==0:
        job_coords = []
    local_job_coords = None
    if r_vals_output is not None:
        path_exists = None
        total_job_size = None
        counts = None
        offsets = None
        if rank==0:
            path_exists = os.path.exists(r_vals_output)
            comm.bcast(path_exists, root=0)
            if path_exists:
                r_vals = -1*np.ones((len(delta_one_arr), len(delta_two_arr)))
                with open(r_vals_output, 'r') as file:
                    text = file.readlines()
                    for line in text:
                        coords, val = tuple(line.split(": "))
                        val = float(val)
                        row, col = tuple(coords.split(" ,"))
                        row = int(row[1:])
                        col = int(col[:-1])
                        r_vals[row, col] = val


                for i in range(len(delta_one_arr)*len(delta_two_arr)):
                    row = i // num_points
                    col = i % num_points
                    if r_vals[row, col] == -1:
                        r_vals[row, col] = 0
                        job_coords.append(i)
                total_job_size = len(job_coords)
                comm.bcast(total_job_size, root=0)
                num_local_points = total_job_size // size
                if rank < total_job_size % size:
                    num_local_points += 1
                start_ind = 0
                if rank < total_job_size % size:
                    start_ind = num_local_points * rank
                else:
                    num_large_ranks = total_job_size % size
                    start_ind = num_large_ranks * (num_local_points + 1) + (rank - num_large_ranks) * num_local_points
                local_vals = np.zeros(num_local_points)
                local_job_coords = np.zeros(num_local_points, dtype=np.int_)
                num_large_ranks = total_job_size % size
                counts = np.array([total_job_size // size + 1]*num_large_ranks + [total_job_size // size]*(size-num_large_ranks), dtype=np.int_)
                if total_job_size // size > 0:
                    offsets = np.array(list(range(0, num_large_ranks*(total_job_size // size + 1), total_job_size // size + 1)) + list(range(num_large_ranks*(total_job_size // size + 1), total_job_size, total_job_size // size)), dtype=np.int_)
                else:
                    offsets = np.array(list(range(0, num_large_ranks * (total_job_size // size + 1), total_job_size // size + 1)) + [total_job_size]*(size-num_large_ranks), dtype=np.int_)
                job_coords = np.array(job_coords, dtype=np.int_)
                comm.Scatterv([job_coords, counts, offsets, MPI.LONG], local_job_coords, root=0)

            else:
                total_job_size = num_points*num_points
                job_coords = list(range(num_points*num_points))
                num_local_points = (num_points * num_points) // size
                if rank < (num_points * num_points) % size:
                    num_local_points += 1
                start_ind = 0
                if rank < (num_points * num_points) % size:
                    start_ind = num_local_points * rank
                else:
                    num_large_ranks = (num_points * num_points) % size
                    start_ind = num_large_ranks * (num_local_points + 1) + (rank - num_large_ranks) * num_local_points
                local_job_coords = list(range(start_ind, start_ind + num_local_points))
                local_vals = np.zeros(num_local_points)

        else:
            path_exists = comm.bcast(path_exists, root=0)
            if path_exists:
                total_job_size = comm.bcast(total_job_size, root=0)
                num_local_points = total_job_size // size
                if rank < total_job_size % size:
                    num_local_points += 1
                start_ind = 0
                if rank < total_job_size % size:
                    start_ind = num_local_points * rank
                else:
                    num_large_ranks = total_job_size % size
                    start_ind = num_large_ranks * (num_local_points + 1) + (rank - num_large_ranks) * num_local_points
                local_vals = np.zeros(num_local_points)
                local_job_coords = np.zeros(num_local_points, dtype=np.int_)
                comm.Scatterv([job_coords, counts, offsets, MPI.LONG], local_job_coords, root=0)
            else:
                num_local_points = (num_points * num_points) // size
                if rank < (num_points * num_points) % size:
                    num_local_points += 1
                start_ind = 0
                if rank < (num_points * num_points) % size:
                    start_ind = num_local_points * rank
                else:
                    num_large_ranks = (num_points * num_points) % size
                    start_ind = num_large_ranks * (num_local_points + 1) + (rank - num_large_ranks) * num_local_points
                local_job_coords = list(range(start_ind, start_ind + num_local_points))
                local_vals = np.zeros(num_local_points)
    else:
        if rank == 0:
            total_job_size = num_points * num_points
            job_coords = list(range(num_points*num_points))
        num_local_points = (num_points * num_points) // size
        if rank < (num_points * num_points) % size:
            num_local_points += 1
        start_ind = 0
        if rank < (num_points * num_points) % size:
            start_ind = num_local_points * rank
        else:
            num_large_ranks = (num_points * num_points) % size
            start_ind = num_large_ranks * (num_local_points + 1) + (rank - num_large_ranks) * num_local_points
        local_job_coords = list(range(start_ind, start_ind+num_local_points))
        local_vals = np.zeros(num_local_points)



#############################################################################################

    if calculate_pairwise_vals:
        local_r_pairs = np.zeros((num_spins, len(local_vals)))
        local_phi_pairs = np.zeros((num_spins, len(local_vals)))

    # This loop runs over the assigned job_ind vals for the current process at which to calculate r_vals
    for job_ind in range(num_local_points):
        ind = local_job_coords[job_ind] # ind is the actual job assignment for the current process
        r_sum = np.longdouble(0) # A running total that will be averaged over trajectories and time values at the end of the loop

        # If the pairwise quantities are being calculated, create a running sum for them as well
        if calculate_pairwise_vals:
            r_pair_sums = np.zeros(num_spins, dtype=np.longdouble)
            phi_sums = np.zeros(num_spins, dtype=np.longdouble)

        # Quantities below are used to calculate the error on the current rms_av value
        r_av = 0
        r_square_sum = np.longdouble(0)
        not_within_fluctuation_tolerance = True

        # Get the grid position of the current job
        row = ind//num_points
        col = ind%num_points

        # Calculate delta_one, delta_two, and delta_three
        delta_one = row * (delta_max - delta_min) / num_points + delta_min
        delta_two = col * (delta_max - delta_min) / num_points + delta_min
        delta_three = -delta_one - delta_two

        # Calculate the full coupling matrix C
        coupling_matrix = coupling_strength * coupling_strength * anti_symmetric_ring_matrix(3) / 2
        coupling_matrix[0, 1] -= delta_one
        coupling_matrix[0, 2] += delta_one
        coupling_matrix[1, 0] += delta_two
        coupling_matrix[1, 2] -= delta_two
        coupling_matrix[2, 0] -= delta_three
        coupling_matrix[2, 1] += delta_three
        coupling_matrix = 1j * coupling_matrix

        # Decompose C into C^h+, C^h-, and C^a as described in Sec. S1.
        coupling_hp, coupling_hm, coupling_a = decompose_coupling_matrix(coupling_matrix)

        # Calculate gamma_g_mod. This is the factor in square brackets in Eq. 8b.
        gamma_g_mod = (spin + 1)/spin * gamma_g
        for k in range(num_spins):
            gamma_g_mod += np.real(coupling_hp[k, k])/spin

        ### Code implementing sde_jit solver ###

        # f is a list of six real functions (three complex functions) corressponding to the drift dynamics in Eq. 8a
        f = []

        # The first three are the real components Re(\mu_1), Re(\mu_2), Re(\mu_3)
        # for k in range(num_spins):
        #     f.append(mu_x(y, k, num_spins, omega, gamma_g_mod, gamma_d, spin, coupling_matrix, epsilon))
        # The final three are the imaginary components Im(\mu_1), Im(\mu_2), Im(\mu_3)
        # for k in range(num_spins):
        #     f.append(mu_y(y, k, num_spins, omega, gamma_g_mod, gamma_d, spin, coupling_matrix, epsilon))

        # Uncomment the block below if you wish to take the classical limit as spin -> infinity instead of the
        # Semiclassical functions. These correspond to Eqs. S4a and S4b under a coordinate change when epsilon = 0.

        for k in range(num_spins):
            f.append(mu_x_classical(y, k, num_spins, omega, gamma_g, gamma_d, coupling_matrix, epsilon))
        for k in range(num_spins):
            f.append(mu_y_classical(y, k, num_spins, omega, gamma_g, gamma_d, coupling_matrix, epsilon))

        # The function g gives the noise terms from Eq. 8c. Note that the real and imaginary components are the same.
        # g = [
        #     p * symengine.sqrt(mat_diff(y(0), y(3), gamma_g, gamma_d, spin)),
        #     p * symengine.sqrt(mat_diff(y(1), y(4), gamma_g, gamma_d, spin)),
        #     p * symengine.sqrt(mat_diff(y(2), y(5), gamma_g, gamma_d, spin)),
        #     p * symengine.sqrt(mat_diff(y(0), y(3), gamma_g, gamma_d, spin)),
        #     p * symengine.sqrt(mat_diff(y(1), y(4), gamma_g, gamma_d, spin)),
        #     p * symengine.sqrt(mat_diff(y(2), y(5), gamma_g, gamma_d, spin))]

        g = [0, 0, 0, 0, 0, 0]

        # Compile the C code and begin the simulation
        SDE = jitcsde(f, g)
        curr_traj = 0
        while curr_traj < total_num_traj and not_within_fluctuation_tolerance:

            # Generates a random initial condition Centered around the stable limit cycle
            initial_amplitudes = np.sqrt(gamma_d/gamma_g) - np.sqrt(gamma_d/gamma_g - 1) + 1e-2*(2*np.random.rand(num_spins)-1)
            initial_phases = 2*np.pi*(np.random.rand()-0.5)*np.ones(num_spins) + 1e-1*2*(np.random.rand(num_spins)-0.5)
            initial_state = np.concatenate((initial_amplitudes*np.cos(initial_phases), initial_amplitudes*np.sin(initial_phases)))
            SDE.set_initial_value(initial_state, 0)

            # Run from t=0 to t=cut_time without recording data
            for time in times[:cut_ind]:
                SDE.integrate(time)

            # Run from t=cut_time to t=t_final while recording data
            for time in times[cut_ind:len(times)]:
                data = SDE.integrate(time)

                # Calculate the current r using the expectation <\vec{z}(t)|R|\vec{z}(t)>
                r = (num_spins - 1) / num_spins
                for k in range(num_spins):
                    mod_squared_k = data[k] * data[k] + data[k + num_spins] * data[k + num_spins]
                    for l in range(k+1, num_spins):
                        mod_squared_l = data[l] * data[l] + data[l + num_spins] * data[l + num_spins]
                        r -= 2 * (4 * (data[k] * data[l] + data[k + num_spins] * data[l + num_spins]) + (
                                    1 - mod_squared_k) * (1 - mod_squared_l)) / (
                                           (1 + mod_squared_k) * (1 + mod_squared_l) * num_spins * num_spins)
                # Add the r to the running r_sum. Also add the square to r_square_sum for error calculation.
                r_sum += r
                r_square_sum += r*r

                # If the pairwise quantities are being calculated as well, this look will calculate them
                if calculate_pairwise_vals:
                    for k in range(num_spins):
                        curr_pair = [0, 1, 2]
                        curr_pair.pop(k)
                        mod_squared_n = data[curr_pair[0]] * data[curr_pair[0]] + data[curr_pair[0] + num_spins] * data[curr_pair[0] + num_spins]
                        mod_squared_m = data[curr_pair[1]] * data[curr_pair[1]] + data[curr_pair[1] + num_spins] * data[curr_pair[1] + num_spins]
                        r_pair = 0.5 - 2 * (4 * (data[curr_pair[0]] * data[curr_pair[1]] + data[curr_pair[0] + num_spins] * data[curr_pair[1] + num_spins])
                                            + (1 - mod_squared_n) * (1 - mod_squared_m)) / ((1 + mod_squared_n) * (1 + mod_squared_m) * 4)

                        phi_sums[k] += np.abs(np.arctan2(data[curr_pair[0]], data[curr_pair[0] + num_spins]) - np.arctan2(data[curr_pair[1]], data[curr_pair[1] + num_spins]))

            # Loop breaks once curr_traj == total_num_traj is reached or r_fluctuation is less than fluctuation_tolerance
            num_samples = (len(times) - cut_ind)*(curr_traj+1)
            r_fluctuation = float(np.sqrt((r_square_sum - r_sum*r_sum/num_samples))/num_samples)
            not_within_fluctuation_tolerance = r_fluctuation > fluctuation_tolerance
            curr_traj += 1

        # Save the r_val into local_vals
        local_vals[job_ind] = float(r_sum/((len(times) - cut_ind)*(curr_traj)))
        # Save the pairwise quantities
        if calculate_pairwise_vals:
            for k in range(num_spins):
                local_r_pairs[k, job_ind] = float(r_pair_sums[k]/((len(times) - cut_ind)*(curr_traj)))
                local_phi_pairs[k, job_ind] = float(phi_sums[k]/((len(times) - cut_ind)*(curr_traj)))
        # Write the current r_val into an output file. This is useful if one expects the simulation time to exceed the
        # alotted time for the simulation and it must be re-started later.
        if r_vals_output is not None:
            with open(r_vals_output, 'a') as file:
                file.write('(' + str(row) + ' ,' + str(col) + '): ' + str(local_vals[job_ind]) + '\n')

    # Once all points in local_job_coords have been looped over, transmit the data back to rank 0.
    gather_arr = None
    gather_pair_arr = None
    gather_phi_arr = None
    temp_arr1 = None
    temp_arr2 = None
    if rank==0:
        gather_arr = np.zeros(total_job_size)
        if calculate_pairwise_vals:
            gather_pair_arr = np.zeros((num_spins, total_job_size))
            gather_phi_arr = np.zeros((num_spins, total_job_size))
            temp_arr1 = np.zeros(len(gather_pair_arr[k, :]))
            temp_arr2 = np.zeros(len(gather_phi_arr[k, :]))
    comm.Gatherv(local_vals, gather_arr, root=0)
    if calculate_pairwise_vals:
        for k in range(num_spins):
            comm.Gatherv(local_r_pairs[k], temp_arr1, root=0)
            comm.Gatherv(local_phi_pairs[k], temp_arr2, root=0)
            if rank==0:
                for l in range(len(temp_arr1)):
                    gather_pair_arr[k, l] = temp_arr1[l]
                    gather_phi_arr[k, l] = temp_arr2[l]
##################################################################################################

    # Rank 0 must now organize all data into the 2D array r_vals.
    if rank==0:
        if calculate_pairwise_vals:
            r_pair_vals = np.zeros((3, num_points, num_points))
            phi_pair_vals = np.zeros((3, num_points, num_points))
        for i in range(len(job_coords)):
            row = job_coords[i]//num_points
            col = job_coords[i]%num_points
            r_vals[row, col] = gather_arr[i]
            if calculate_pairwise_vals:
                for k in range(num_spins):
                    r_pair_vals[k, row, col] = gather_pair_arr[k, i]
                    phi_pair_vals[k, row, col] = gather_phi_arr[k, i]

        # Plot the r_vals
        plt.figure()
        plt.pcolormesh(
            np.reshape(np.kron(np.ones(len(delta_one_arr)), delta_one_arr), (len(delta_one_arr), len(delta_one_arr))),
            np.reshape(np.kron(delta_two_arr, np.ones(len(delta_two_arr))), (len(delta_two_arr), len(delta_two_arr))),
            r_vals, shading='nearest')
        plt.title(r"r-Parameter Values")
        plt.xlabel(r"$\Delta_1$")
        plt.ylabel(r"$\Delta_2$")
        plt.colorbar(orientation='vertical', ax=plt.gca())
        plt.savefig("scaledRMS_vs_delta_spin_{}_mpi.png".format(spin))

        # Save all data to .csv files.
        np.savetxt("delta_vals_mpi.csv", delta_one_arr, delimiter=',')
        np.savetxt("RMS_vals_mpi.csv", r_vals, delimiter=',')
        if calculate_pairwise_vals:
            for k in range(num_spins):
                np.savetxt("Pairwise_RMS_{}.csv".format(k), r_pair_vals[k, :, :], delimiter=',')
                np.savetxt("Pairwise_phi_{}.csv".format(k), phi_pair_vals[k, :, :], delimiter=',')



