import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from qutip import *
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
import cupy as cp
from cupy_backends.cuda.libs.cusparse import CuSparseError
from cupy.cuda.memory import OutOfMemoryError
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import eye as cp_eye
from scipy.sparse import csr_array, eye, kron, save_npz

# Generates the adjacency matrix of an all2all graph
def all2all_adjacency_matrix(N):
    """
    Generates the adjacency matrix of an all2all graph

    :param N: Number of nodes
    :return: Adjacency matrix of a complete graph
    """

    graph = np.ones((N, N)) - np.diag(np.ones(N))
    return graph

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

def construct_spin_network_gen_coupling(spin, omega_list, coupling_matrix, damping_rates, gain_rates):
    """
    Generates the Hamiltonian in Eq. S2a and a list of jump operators found in Eqs. S2b and S2c in the Supplemental.
    Note that qtip does not directly support the two argument dissipators found in Eq. S2c, so these are instead converted
    into single argument dissapators

    :param spin: Spin quantum number
    :param omega_list: List of omegas
    :param coupling_matrix: Coupling matrix between the spins
    :param damping_rates: The individual non-linear damping rate for each spin
    :param gain_rates: The individual linea gain rate for each spin
    :return: total_ham which is the total hamiltonian for the system and c_ops, the full list of collapse operators for
    the system.
    """

    # Declare variables which will be used throughout function
    sz = spin_Jz(spin)
    s_plus = spin_Jp(spin)
    num_oscillators = len(omega_list)
    if num_oscillators != len(coupling_matrix) or num_oscillators != len(damping_rates) or num_oscillators != len(
            gain_rates):
        raise ValueError('Invalid input')
    c_ops = []
    dims = int(2 * spin + 1)

    # Create a Hamiltonian of all zeros
    total_ham = Qobj(np.zeros((dims ** num_oscillators, dims ** num_oscillators)),
                     dims=[[dims] * num_oscillators, [dims] * num_oscillators])

    # Creates C^h and C^a matrices from section S1.
    coupling_herm = 0.5 * (coupling_matrix + np.conjugate(coupling_matrix).T)
    coupling_anti_herm = 0.5 * (coupling_matrix - np.conjugate(coupling_matrix).T)

    # Finds the eigenvalues and eigenvecs of C^h. These will be used to create the dissipators of Eq. S2c.
    if len(coupling_matrix) > 1:
        eig_vals, eig_vecs = np.linalg.eigh(coupling_herm)
    else:
        eig_vals = [0]
    for i in range(num_oscillators):
        # Add the the term ωS^z_i to total_ham
        total_ham += omega_list[i] * tensor([qeye(dims)] * i + [sz] + [qeye(dims)] * (
                    num_oscillators - i - 1))

        # Create the raising operator for the ith spin
        plus_i = tensor([qeye(dims)] * i + [s_plus] + [qeye(dims)] * (num_oscillators - i - 1))
        for j in range(i + 1, num_oscillators):
            # Add the hamiltonian portion of the coupling between spins i and j in Eq. S2C to total_ham
            minus_i_plus_j = tensor(
                [qeye(dims)] * i + [s_plus.dag()] + [qeye(dims)] * (j - i - 1) + [s_plus] + [qeye(dims)] * (
                            num_oscillators - j - 1))
            total_ham += 0.5j * (coupling_anti_herm[i, j] * minus_i_plus_j - np.conjugate(
                coupling_anti_herm[i, j]) * minus_i_plus_j.dag())

        # If the ith eigenvalue of C^h is nonzero, a jump operator is created. It is a weighted sum of lowering
        # (raising) operators with given that the ith eigenvalue is positive (negative). The weights given by the ith
        # eigenvector for a positive eigenvalue, or the complex conjugate of the eigenvalue if the weights are negative.
        # The sum over the range of i of these jump operators is equal to the two argument dissipators in Eq. S2c.
        if (not np.isclose(eig_vals[i], 0)):
            c_op_current = Qobj(np.zeros((dims ** num_oscillators, dims ** num_oscillators)),
                                dims=[[dims] * num_oscillators, [dims] * num_oscillators])
            if eig_vals[i] > 0:
                for j in range(num_oscillators):
                    c_op_current += eig_vecs[j, i] * tensor(
                        [qeye(dims)] * j + [s_plus.dag()] + [qeye(dims)] * (num_oscillators - j - 1))
                c_ops.append(np.sqrt(eig_vals[i]) * c_op_current)
            else:
                for j in range(num_oscillators):
                    c_op_current += np.conjugate(eig_vecs[j, i]) * tensor(
                        [qeye(dims)] * j + [s_plus] + [qeye(dims)] * (num_oscillators - j - 1))
                c_ops.append(np.sqrt(-eig_vals[i]) * c_op_current)

        # Now constuct the damping and gain operators of the ith oscillators found in Eq. S2b.
        damping_op = np.sqrt(damping_rates[i] / (2 * spin * spin)) * plus_i * plus_i
        gain_op = np.sqrt(gain_rates[i]) * plus_i.dag()
        c_ops.append(damping_op)
        c_ops.append(gain_op)
    return total_ham, c_ops

def spin_variance_operator(spin, num_spins):
    """
    Constructs a Qobj that corresponds to the operator R is defined in Eq. S9 and can be equivalently defined as
    R = S(S+1) - (1/N^2)(S_tot)^2.

    :param spin: The spin quantum number
    :param num_spins: The number of spins in the system
    :return: A qobj that corresponds to the operator R in Eq. 7 in the main article
    """
    dimensions = int(2 * spin + 1)
    raising_op_single = spin_Jp(spin)
    z_op_single = spin_Jz(spin)

    # First set operator R equal to S(S+1)
    variance_op = spin*(spin+1)*(num_spins - 1)*qeye([dimensions]*num_spins)/num_spins

    # The followinging two loops will subtract (1/N^2)(S_tot)^2 from the initial value of R above. The double loop
    # results from expanding (S_tot)^2 = ||\sum_n=1^N \vec{S}_n ||^2 in terms of individual spin operators.
    for i in range(num_spins):
        current_raising_op_left = tensor([qeye(dimensions)]*i + [raising_op_single] + [qeye(dimensions)]*(num_spins - i - 1))
        current_z_op_left = tensor([qeye(dimensions)]*i + [z_op_single] + [qeye(dimensions)]*(num_spins - i - 1))
        for j in range(i+1, num_spins):
            current_raising_op_right = tensor([qeye(dimensions)]*j + [raising_op_single] + [qeye(dimensions)]*(num_spins - j - 1))
            current_z_op_right = tensor([qeye(dimensions)]*j + [z_op_single] + [qeye(dimensions)]*(num_spins - j - 1))
            phase_coherence = current_raising_op_left * current_raising_op_right.dag()
            variance_op -= 2 * (0.5 * (phase_coherence + phase_coherence.dag()) + current_z_op_left * current_z_op_right)/(num_spins * num_spins)
    return variance_op

def get_sparse_liouvillian(hamiltonian, c_ops, format="csr"):
    """
    This function returns a scipy sparse matrix for the liouvillian give a hamiltonian and list of collapse operators.
    Note that this assumes that the density matrix is in row major order. Qutip functions assume that the density matrix
    superoperator is in column major ordering. To convert the Liouvillian to column major, take the complex conjugate of
    the row major Liouvillian and likewise for column to row major format.
    :param hamiltonian: A qobj which is the hamiltonian of the full system
    :param c_ops: A list containing all of the collapse operators as Qobjs of the full system.
    :param format: An N^2 x N^2 sparse matrix corresponding to the liouvillian in superoperator form
    :return: The system liouvillian superoperator in sparse format
    """
    #Find Hamiltonian commutator to get the Schrodinger evolution of system
    dense_matrix = hamiltonian.full()
    hilbert_dims = len(dense_matrix)
    sparse_op = csr_array(dense_matrix)
    sparse_idty = eye(hilbert_dims, format='csr')
    liouvillian = -1j*(kron(sparse_op, sparse_idty)-kron(sparse_idty, sparse_op.conjugate()))
    #Construct the dissapators for each c_op
    for c_op in c_ops:
        dense_matrix = c_op.full()
        dense_herm = np.conjugate(dense_matrix.T)@dense_matrix
        sparse_op = csr_array(dense_matrix)
        liouvillian += kron(sparse_op, sparse_op.conjugate())
        sparse_op = csr_array(dense_herm)
        liouvillian -= 0.5*(kron(sparse_op, sparse_idty) + kron(sparse_idty, sparse_op.T))
    if format=="csc":
        liouvillian = liouvillian.tocsc()
    elif format=="coo":
        liouvillian = liouvillian.tocoo()
    return liouvillian

def get_max_iter(spin, omega, gamma_g, gamma_d, coupling_strength, delta_min, delta_max, num_points):
    """
    Constructs the liouvillian (denoted as L) corresponding to Eq. S1 in superoperator form and calculates the maximum
    power the operator exp_liouvillian = (1 - final_t_to_n_ratio*L) before the program crashes due to overallocation of
    VRAM. The return value of max_iter will be used in the function get_r_gpu to ensure the function will not crash due
    to VRAM overallocation when it is ran during the main program.

    :param spin: The spin quantum number of the system. All oscillators are assumed to have the same spin.
    :param omega: The natural precession frequency of the system. Is equal for all oscillators
    :param gamma_g: Individual gain of the three oscillators
    :param gamma_d: Individual damping of the three oscillators
    :param coupling_strength: Individual coupling strength of the three oscillators
    :param delta_min: Minimum value of delta_1 and delta_2.
    :param delta_max: Maximum value of delta_1 and delta_2.
    :param num_points: The number of points in delta_one_arr and delta_two_arr
    :return: Returns max_iter, which is maximum k such that (1 - final_t_to_n_ratio*L)^(2^k) does not crash due to
    memory overallocation.
    """

    # Initialize all parameters as one would in the function get_r_gpu
    max_iter = 0
    # Here it is assumed that all values of row and col will have the same value of max_iter. Therefore one may set
    # row = 0 and col = 0
    row = 0
    col = 0
    delta_one = row * (delta_max - delta_min) / num_points + delta_min
    delta_two = col * (delta_max - delta_min) / num_points + delta_min
    delta_three = -delta_one - delta_two

    coupling_matrix = coupling_strength * coupling_strength * anti_symmetric_ring_matrix(3) / 2

    # Assumes that each spin has the same gain
    gamma_g_arr = [gamma_g] * 3

    # Set up coupling matrix
    coupling_matrix[0, 1] -= delta_one
    coupling_matrix[0, 2] += delta_one
    coupling_matrix[1, 0] += delta_two
    coupling_matrix[1, 2] -= delta_two
    coupling_matrix[2, 0] -= delta_three
    coupling_matrix[2, 1] += delta_three
    coupling_matrix = 1j * coupling_matrix

    # Set up omega_arr
    omega_arr = [omega] * 3
    opr_len = int(2 * spin + 1) ** 6

    # Set up the liouvillan of the full system as a sparse matrix
    sparse_liouvillian = get_sparse_liouvillian(*construct_spin_network_gen_coupling(spin, omega_arr,
                                                                                     coupling_matrix, [gamma_d] * 3,
                                                                                     gamma_g_arr))
    # Guess for how long it takes for the system to reach a steady state.
    final_time_guess = 10
    # Smaller values of this will give more accurate results, but take longer to compute. Note that for larger values of
    # this parameter the power method iteration will diverge, so you may need to take multiple test runs at various
    # final_t_to_n_ratio values.
    final_t_to_n_ratio = 1e-03
    # This is the maximum number of times dev_exp_liouvillian will need to be multiplied by itself to reach the taget
    # time
    num_iterations = int(np.ceil(np.log2(final_time_guess / final_t_to_n_ratio)))

    # dev_exp_liouvillian equals (1 - liouvillian * t/n) where t/n is the same as final_t_to_n_ratio. dev_exp_liouvillian
    # will be multiplied by itself either until dev_exp_liouvillian equals (1 - liouvillian * t/n)^n or VRAM runs out.
    # The final value of max_iter will be log_2(m) where m <= n is the maximum actual achievable matrix power.
    dev_exp_liouvillian = (cp_eye(opr_len, dtype=cp.complex128, format='csr') + cp_csr_matrix(
        (cp.array(sparse_liouvillian.data),
         cp.array(sparse_liouvillian.indices),
         cp.array(sparse_liouvillian.indptr)), shape=(opr_len, opr_len),
        dtype=cp.complex128) * cp.float64(final_t_to_n_ratio))
    try:
        while max_iter < num_iterations:
            dev_exp_liouvillian @= dev_exp_liouvillian
            max_iter += 1
    except (CuSparseError, OutOfMemoryError):
        return max_iter
    return max_iter

def get_r_gpu(params, rank_assignments):
    """
    Returns the value of r from Eq. 7 for each point in the delta_1-delta_2 plane as specified by rank_assignments. \
    :param params: The static parameters which do not change between gridpoints
    :param rank_assignments: The list of all grid points that are desired to be calculated
    :return: the r_value and accompanying gpu_rank which calculated it.
    """
    gpu_rank, job_coords = rank_assignments

    # Assign the gpu with gpu_rank to to the current CPU.
    with cp.cuda.Device(gpu_rank):
        spin, omega, gamma_g, gamma_d, coupling_strength, delta_min, delta_max, num_points, max_iter, num_gpus, r_vals_output = params
        r_vals = np.zeros(len(job_coords)) # Stores values of r local to this process
        tolerance = 1e-16
        for i in range(len(job_coords)):
            # Get the row and column on the grid which corresponds the index in job_coords
            ind = job_coords[i]
            row = ind // num_points
            col = ind %  num_points

            # Get the values of the three deltas based on the values of row and col
            delta_one = row * (delta_max - delta_min) / num_points + delta_min
            delta_two = col * (delta_max - delta_min) / num_points + delta_min
            delta_three = -delta_one - delta_two

            # Construct the coupling matrix from Eq. 6 in the main text
            coupling_matrix = coupling_strength * coupling_strength * anti_symmetric_ring_matrix(3) / 2
            coupling_matrix[0, 1] -= delta_one
            coupling_matrix[0, 2] += delta_one
            coupling_matrix[1, 0] += delta_two
            coupling_matrix[1, 2] -= delta_two
            coupling_matrix[2, 0] -= delta_three
            coupling_matrix[2, 1] += delta_three
            coupling_matrix = 1j * coupling_matrix

            # Assign internal parameters of each node from Eqs. S2a and S2b.
            gamma_g_arr = [gamma_g] * 3
            gamma_d_arr = [gamma_d] * 3
            omega_arr = [omega] * 3

            sparse_liouvillian = get_sparse_liouvillian(*construct_spin_network_gen_coupling(spin, omega_arr,
                                                                                             coupling_matrix, gamma_d_arr,
                                                                                             gamma_g_arr))

            opr_len = int(2 * spin + 1) ** 6
            dev_sparse_liouvillian = cp_csr_matrix((cp.array(sparse_liouvillian.data), cp.array(sparse_liouvillian.indices),
                 cp.array(sparse_liouvillian.indptr)), shape=(opr_len, opr_len), dtype=cp.complex128)

            ###Attempt to construct steady state projector using exponential of Liouvillian###
            final_time_guess = 10
            final_t_to_n_ratio = 1e-3  # Smaller values of this will give more accurate results, but take longer to compute
            num_iterations = int(np.ceil(np.log2(final_time_guess / final_t_to_n_ratio)))

            # dev_exp_liouvillian equals (1 - liouvillian * t/n) where t/n is the same as final_t_to_n_ratio. dev_exp_liouvillian
            # will be multiplied by itself either until dev_exp_liouvillian equals (1 - liouvillian * t/n)^n or VRAM runs out.
            # Since the spectrum of dev_exp_liouvillian is bounded between 1 and 0, ideally only the eigenvector with
            # eigenvalue 1 (the steady state) will remain after the repeated multiplication.
            dev_exp_liouvillian = (cp_eye(opr_len, dtype=cp.complex128, format='csr') + dev_sparse_liouvillian * cp.float64(final_t_to_n_ratio))
            counter = 0
            keep_looping = counter == max_iter
            while(keep_looping):
                try:
                    dev_exp_liouvillian @= dev_exp_liouvillian
                    counter += 1
                    keep_looping = counter == max_iter
                except (CuSparseError, OutOfMemoryError):
                    keep_looping = False

            # If counter is not equal to num_iteratoins, then the remaining number of iterations needed will be calculated
            # by using the power method
            num_repeats = 2 ** (num_iterations - counter)
            dev_result = cp.zeros((int(2 * spin + 1) ** 6, 1), dtype=cp.complex128)
            dev_result[0] = cp.complex128(1)
            second_loop_iterations = 0
            while not cp.allclose(dev_sparse_liouvillian @ dev_result, cp.zeros((int(2 * spin + 1) ** 6, 1), dtype=cp.complex128), atol=tolerance).get():
                if second_loop_iterations >2:
                    print("Help! I'm stuck in a loop, starting iteration {}. Steady state norm is {}".format(second_loop_iterations, cp.linalg.norm(dev_result)))
                    sys.stdout.flush()
                counter = 0
                second_loop_iterations += 1
                while counter < num_repeats:
                    dev_result = dev_exp_liouvillian @ dev_result
                    counter += 1

            # Calculate r based on the expectation value of the steady state
            rescaled_r_op = (spin_variance_operator(spin, 3) - (2 * spin / 3) *
                               qeye([int(2 * spin) + 1] * 3)) / (spin * (spin + 1) - (2 * spin / 3))
            rho_ss = Qobj(dev_result.get().reshape((int(2 * spin) + 1) ** 3, (int(2 * spin) + 1) ** 3),
                          dims=[[int(2 * spin) + 1] * 3, [int(2 * spin) + 1] * 3])
            rho_ss = rho_ss / rho_ss.tr()
            # !!! Saving the steady-states is optional and can take significant disk space, so you may comment out this
            # line if desired
            save_npz("states/steady_state_{}_{}_{}.npz".format(spin, row, col), csr_array(rho_ss.full()))

            # Save the value of r and print it out to r_vals_output to save progress
            r = np.real((rho_ss * rescaled_r_op).tr())
            r_vals[i] = r
            if r_vals_output is not None:
                with open(r_vals_output, 'a') as file:
                    file.write('(' + str(row) + ' ,' + str(col) + '): ' + str(r) + '\n')
            sys.stdout.flush()
    return r_vals, gpu_rank

if __name__ == '__main__':

    # This ensures that the processes launched later in the script can properly interface with the GPUs
    multiprocessing.set_start_method('spawn')
    spin = 6  # Spin quantum number
    omega = -1  # Precession frequency about the applied external field
    gamma_d = 2  # Damping strength which corresponds to quadrupole coupling to thermal fluctuations
    gamma_g = 1  # Gain strength which corresponds to dipole coupling to external field
    coupling_strength = 1 # Scalar factor that multiplies the coupling matrix C

    delta_max = 2*coupling_strength # Maximum value of Δ_1 and Δ_2
    delta_min = -2*coupling_strength # Minimum value of Δ_1 and Δ_2
    grid_step = 0.01 * (delta_max - delta_min) # The step between points in the delta_min to delta_max interval
    num_points = int(np.floor((delta_max - delta_min)/grid_step))
    delta_one_arr = np.linspace(delta_min, delta_max, num_points) # Construction of the delta_one_arr
    delta_two_arr = np.linspace(delta_min, delta_max, num_points) # Construction of the delta_two_arr
    r_vals = np.zeros((len(delta_one_arr), len(delta_two_arr))) # Empty grid to contain the values of r


    num_gpus = cp.cuda.runtime.getDeviceCount()
    # Determines the maximum power to which the propagator generated by the Liouvillian can be raised. Used in power method calculation
    max_iter = get_max_iter(spin, omega, gamma_g, gamma_d, coupling_strength, delta_min, delta_max, num_points)

    # If this simulation is a continuation of a previous one, enter in the name of the grid_vals_InsertSpin#.out as the
    # r_vals_output file.
    spin_str = None
    if isinstance(spin, int):
        spin_str = str(spin)
    else:
        spin_str = str(spin).split('.')
        spin_str = spin_str[0] + ',' + spin_str[1]
    # r_vals_output = None
    r_vals_output = "grid_vals_" + spin_str + ".out"
    job_coords = []


    # This reads in the data stored in the previous r_vals_output file to continue the previous calculation
    if r_vals_output is not None:
        if os.path.exists(r_vals_output):
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

            for i in range(num_points):
                for j in range(num_points):
                    if r_vals[i, j] == -1:
                        r_vals[i, j] = 0
                        job_coords.append(i*num_points + j)
        else:
            job_coords = np.array(range(num_points * num_points), dtype=int)
    else:
        job_coords = np.array(range(num_points*num_points), dtype=int)

    # Assigns each process an equal number of points from job_coords on which to calculate r_vals.
    rank_assignments = []
    for gpu_rank in range(num_gpus):
        arr_len = len(job_coords)//num_gpus
        extra_points = len(job_coords)%num_gpus
        start = gpu_rank * arr_len
        if gpu_rank<extra_points:
            start = start + gpu_rank
            arr_len = arr_len + 1
        else:
            start = start + extra_points
        rank_inds = np.zeros(arr_len, dtype=int)
        for i in range(arr_len):
            rank_inds[i] = job_coords[start+i]
        rank_assignments.append((gpu_rank, rank_inds))


    # Params is a tuple of all the static parameters that are constant for all values in job_coords.
    params = (spin, omega, gamma_g, gamma_d, coupling_strength, delta_min, delta_max, num_points, max_iter, num_gpus, r_vals_output)
    # Starts a multiprocessing pool with num_gpus number of processes. Each process interfaces with a single gpu
    with Pool(processes=num_gpus) as pool:
        for ind, res in enumerate(pool.starmap(get_r_gpu, zip(repeat(params), rank_assignments))):
            local_points, gpu_rank = res
            for i in range(len(local_points)):
                row = rank_assignments[gpu_rank][1][i]//num_points
                col = rank_assignments[gpu_rank][1][i]%num_points
                r_vals[row, col] = local_points[i]

    # Create figure and save the r_vals in a .csv file
    plt.figure()
    plt.pcolormesh(
        np.reshape(np.kron(np.ones(len(delta_one_arr)), delta_one_arr), (len(delta_one_arr), len(delta_one_arr))),
        np.reshape(np.kron(delta_two_arr, np.ones(len(delta_two_arr))), (len(delta_two_arr), len(delta_two_arr))),
        r_vals, shading='nearest')
    plt.title(r"Synchronization Measure r")
    plt.xlabel(r"$\Delta_1$")
    plt.ylabel(r"$\Delta_2$")
    plt.colorbar(orientation='vertical', ax=plt.gca())
    plt.savefig("r_vs_delta_spin_{}_gpu.png".format(spin_str))

    np.savetxt("delta_vals_{}_gpu.csv".format(spin_str), delta_one_arr, delimiter=',')
    np.savetxt("r_vals_{}_gpu.csv".format(spin_str), r_vals, delimiter=',')
