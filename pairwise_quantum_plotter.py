import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.sparse import csr_array, csr_matrix, load_npz, save_npz
from qutip import *
from math import sqrt
from math import factorial
from scipy.special import gamma
from numba import jit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import matplotlib.gridspec as gridspec

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


def relative_phase_distribution2(rho, subsys_a, subsys_b, phi_dist, matrix_elements=None):
    """
    Wrapper function for relative_phase_distribution_numba2. This function calculates the entries of matrix_elements
    which employs the gamma function in scipy.special and is not supported by numba's @jit decorator.

    :param rho: Qobj that represents a density matrix of at least two spins.
    :param subsys_a: The index of the first spin's subspace
    :param subsys_b: The index of the second spin's subspace
    :param phi_dist: The values of phi at which to calculate \Phi_{i, j}.
    :param matrix_elements: Elements related to the Wigner D matrix that are used repeatedly throughout the computation
    :return: Values of the distribution S_{ij}(\phi) as shown below Eq. 9 in the main text.
    """
    coo_arr = rho.ptrace([subsys_a, subsys_b]).data.as_scipy().tocoo()
    dims = rho.dims[0][0]
    if matrix_elements is None:
        matrix_elements = np.zeros((dims, dims))
        for i in range(dims):
            for j in range(dims):
                if i == j:
                    matrix_elements[i, j] = 0
                else:
                    matrix_elements[i, j] = ((2 * spin + 1) / (4 * np.pi)) * ((2 * factorial(dims - 1) * gamma(
                        1 + int(2 * spin) - 0.5 * (i + j)) * gamma(1 + 0.5 * (i + j))) /
                                                                              (sqrt(factorial(dims - 1 - i) * factorial(
                                                                                  i) * factorial(dims - 1 - j) * factorial(
                                                                                  j)) * factorial(dims)))
    return relative_phase_distribution_numba2(coo_arr.data, coo_arr.coords, dims, phi_dist, matrix_elements)

@jit
def relative_phase_distribution_numba2(data, coords, dims, phi_dist, matrix_elements):
    """
    Returns an nd.array of real double values whose values are the relative phase distribtuion S_{ij}(\phi) as defined in https://arxiv.org/abs/2208.12766

    input:
    rho - A Qobj that represents a system of n coupled spins.
    subsys_a - index of the first subsystem with which will be measured by the relative phase distribution
    subsys_b - index of the second subsystem which will be measured by the relative phase distribution
    stepsize - The stepsize of phi in the relative phase distribution

    returns:
    An ndarray of doubles whose values correspond to S_{ij}(\phi) where phi ranges from -pi to pi
    """

    rows, cols = coords
    spin = 0.5 * (dims - 1)
    non_zero_inds = []
    for data_ind in range(len(data)):
        n = rows[data_ind]
        m = cols[data_ind]
        m_a = n // dims
        n_a = m // dims
        if n_a > m_a:
            m_b = m % dims
            if m_b < dims + m_a - n_a:
                n_b = n % dims
                if n_a - n_b == m_a - m_b:
                    non_zero_inds.append(data_ind)
    phase_dist = np.zeros((len(phi_dist)))
    # Now calculate the phi distribution
    for i in range(len(phi_dist)):
        phi_current = phi_dist[i]
        for data_ind in non_zero_inds:
            n = rows[data_ind]
            m = cols[data_ind]
            m_a = n // dims
            n_a = m // dims
            m_b = m % dims
            n_b = n % dims
            phase_dist[i] += (4 * np.pi * np.real(np.exp(1j * (n_a - m_a) * phi_current) * data[data_ind]) *
                              matrix_elements[m_a, n_a] * matrix_elements[m_b, n_b])
    return phase_dist

if __name__ == "__main__":
    #Set up all constant parameters throughout calculation
    num_spins = 3
    spin = 6
    coupling_strength = 1

    delta_max = 2*coupling_strength
    delta_min = -2*coupling_strength
    grid_step = 0.01 * (delta_max - delta_min)
    num_points = int(np.floor((delta_max - delta_min)/grid_step))
    delta_one_arr = np.linspace(delta_min, delta_max, num_points)
    delta_two_arr = np.linspace(delta_min, delta_max, num_points)

    # Set up empty lists to store data
    r_vals_pair = np.zeros(((num_spins*(num_spins-1))//2, len(delta_one_arr), len(delta_two_arr)))
    phi_diffs = np.zeros((num_spins, len(delta_one_arr), len(delta_two_arr)))
    diff_sums = np.zeros((len(delta_one_arr), len(delta_two_arr)))
    r_total = np.zeros((len(delta_one_arr), len(delta_two_arr)))
    phi_dist = np.arange(-np.pi, np.pi, 1e-2)

    # Set up operators and matrix_elements which are used in the rms and phi_diff calculations resepectively.

    rescaled_r_op = (spin_variance_operator(spin, 2) - (spin / 2) *
                       qeye([int(2 * spin) + 1] * 2)) / (spin * (spin + 1) - (spin / 2))

    dims = int(2*spin + 1)
    matrix_elements = np.zeros((dims, dims))
    for i in range(dims):
        for j in range(dims):
            if i == j:
                matrix_elements[i, j] = 0
            else:
                matrix_elements[i, j] = ((2 * spin + 1) / (4 * np.pi)) * ((2 * factorial(dims - 1) * gamma(
                    1 + int(2 * spin) - 0.5 * (i + j)) * gamma(1 + 0.5 * (i + j))) /
                                                                          (sqrt(factorial(dims - 1 - i) * factorial(
                                                                              i) * factorial(dims - 1 - j) * factorial(
                                                                              j)) * factorial(dims)))

    # Set up double loop to go through each delta_one_arr-delta_two_arr pair. num_points should match the value of
    # the num_points used in spin_r_vs_delta_gpu_parallel.py when originally calculating the data.
    for i in range(num_points):
        print(i)
        for j in range(num_points):
            # Steady states should be saved as scipy sparse matrices from the spin_r_vs_delta_gpu_parallel.py script.
            # The title of each file should be "steady_state_{spin}_{row}_{col}.npz" so they can be efficiently looped
            # through.
            file_path = "C:\\path\\to\\steady_state\\npz_files\\steady_state_{}_{}_{}.npz".format(spin, i, j)
            rho_ss = Qobj(csr_matrix(load_npz(file_path)), dims=[[2*spin+1]*num_spins,[2*spin+1]*num_spins], copy=False)
            diff_ind = 0
            for spin_a in range(num_spins-1):
                for spin_b in range(spin_a+1, num_spins):
                    # Find the pairwise r_{ij} values
                    r_vals_pair[diff_ind, i, j] = np.real((rescaled_r_op * rho_ss.ptrace([spin_a, spin_b])).tr())

                    phi_probs = relative_phase_distribution2(rho_ss, spin_a, spin_b, phi_dist, matrix_elements=matrix_elements)
                    # Get the \Phi_{ij} values based on Eq. 9 in the main text
                    phi_diffs[diff_ind, i, j] = phi_dist[np.argmax(phi_probs)]
                    diff_ind += 1
            # Find the total r value for all three spins
            r_total[i, j] = ((np.sum(r_vals_pair[:, i, j])*(spin * (spin + 1) - (spin / 2)) + (spin / 2)) - (2 * spin / 3))/(spin * (spin + 1) - (2 * spin / 3))
            # Sum the three \Phi_{ij} values together to get \Phi{sum}
            diff_sums[i, j] = np.sum(np.abs(phi_diffs[:, i, j]))

    # Save all data to output files
    np.savetxt("pairwise_r_spin_{}.csv".format(spin), np.reshape(r_vals_pair, (((num_spins*(num_spins-1))//2)*num_points*num_points)),delimiter=',')
    np.savetxt("total_r_spin_{}.csv".format(spin), r_total, delimiter=',')
    np.savetxt("phi_diffs_spin_{}.csv".format(spin), np.reshape(phi_diffs, (num_spins*num_points*num_points)),delimiter=',')
    np.savetxt("sum_diffs_spin_{}.csv".format(spin), diff_sums, delimiter=',')

    ### Plot data ###

    figsize = 5
    dpi = 100  # 72.035 # 72.05
    fontsize = 20
    fig = plt.figure(figsize=(3.7 * figsize, 3.8 * figsize))  # , dpi=dpi)#, layout='constrained')
    scale = 100  # 100
    offset = 13  # 13
    offset_rows = 10  # 5
    gs = gridspec.GridSpec(nrows=3 * scale - offset_rows, ncols=3 * scale, figure=fig, hspace=0.5,
                           wspace=0.5)  # , wspace=0.4)
    ax_label_scale = 2.5
    ax_ticks_scale = 2
    cbar_scale = 1.8
    cbar_rms_title_scale = 2
    cbar_angle_title_scale = 2
    angle_pad = 30
    angle_shift = 1
    r_pad = angle_pad  # 17
    subfig_label_scale = 2

    equal_aspect = True
    colormap = 'viridis'
    axs_rms = []
    rms_pair_min = np.min(r_vals_pair)
    rms_pair_max = np.max(r_vals_pair)
    norm = mpl.colors.Normalize(rms_pair_min, rms_pair_max)
    for i in range(num_spins):
        curr_ax = fig.add_subplot(gs[scale - offset + offset_rows:2 * (scale - offset), scale * i:scale * (i + 1)])
        axs_rms.append(curr_ax)
        im = curr_ax.pcolormesh(
            np.reshape(np.kron(np.ones(len(delta_one_arr)), delta_one_arr), (len(delta_one_arr), len(delta_one_arr))),
            np.reshape(np.kron(delta_two_arr, np.ones(len(delta_two_arr))), (len(delta_two_arr), len(delta_two_arr))),
            r_vals_pair[i], cmap=colormap, shading='nearest', norm=norm, rasterized=True)
        curr_ax.set_xlabel(r"$\Delta_1$", fontsize=ax_label_scale * fontsize)
        curr_ax.tick_params(axis='x', labelsize=ax_ticks_scale * fontsize)
        if i > 0:
            curr_ax.get_yaxis().set_ticks([])
        else:
            curr_ax.set_ylabel(r"$\Delta_2$", fontsize=ax_label_scale * fontsize)
            curr_ax.tick_params(axis='y', labelsize=ax_ticks_scale * fontsize)
        if equal_aspect:
            curr_ax.set_aspect('equal')  # , adjustable='box')
        if i == num_spins - 1:
            ax_ins = inset_axes(curr_ax, width="5%", height="80%", loc='lower right',
                                # bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = curr_ax.transAxes,
                                bbox_to_anchor=(0.1, 0., 1, 1), bbox_transform=curr_ax.transAxes,
                                # bbox_transform = ax8s.transAxes,
                                borderpad=0)
            cbar_ticks = [rms_pair_min, 0.5 * (rms_pair_max + rms_pair_min), rms_pair_max]
            cbar = plt.colorbar(im, cax=ax_ins, orientation='vertical', norm=norm, ticks=cbar_ticks)
            cbar.ax.set_yticklabels([f"{cbar_ticks[i]:.2f}" for i in range(len(cbar_ticks))])
            cbar.ax.tick_params(labelsize=cbar_scale * fontsize)
            ax_ins.set_title(r"$r_{ij}$", fontsize=cbar_rms_title_scale * fontsize, position=(1.3, 0), pad=r_pad)
    norm = mpl.colors.Normalize(0, np.pi)
    colormap = plt.get_cmap('hsv')
    axs_phase = []
    for i in range(num_spins):
        curr_ax = fig.add_subplot(gs[0:scale - offset - offset_rows, scale * i:scale * (i + 1)])
        axs_phase.append(curr_ax)
        im = curr_ax.pcolormesh(
            np.reshape(np.kron(np.ones(len(delta_one_arr)), delta_one_arr), (len(delta_one_arr), len(delta_one_arr))),
            np.reshape(np.kron(delta_two_arr, np.ones(len(delta_two_arr))), (len(delta_two_arr), len(delta_two_arr))),
            np.abs(phi_diffs[i]), cmap='magma', shading='nearest', norm=norm, rasterized=True)
        curr_ax.get_xaxis().set_ticks([])
        if i > 0:
            curr_ax.get_yaxis().set_ticks([])
        else:
            curr_ax.set_ylabel(r"$\Delta_2$", fontsize=ax_label_scale * fontsize)
            curr_ax.tick_params(axis='y', labelsize=ax_ticks_scale * fontsize)
        if equal_aspect:
            curr_ax.set_aspect('equal')
        if i == num_spins - 1:
            ax_ins = inset_axes(curr_ax, width="5%", height="80%", loc='lower right',
                                bbox_to_anchor=(0.1, 0., 1, 1), bbox_transform=curr_ax.transAxes,
                                borderpad=0)
            cbar_ticks = [0, np.pi / 2, np.pi]
            cbar = plt.colorbar(im, cax=ax_ins, orientation='vertical', norm=norm, ticks=cbar_ticks)
            cbar.ax.set_yticklabels([f"{cbar_ticks[i]:.2f}" for i in range(len(cbar_ticks))])
            cbar.ax.tick_params(labelsize=cbar_scale * fontsize)
            ax_ins.set_title(r"$\Phi_{ij}$", fontsize=cbar_angle_title_scale * fontsize, position=(1.3, 0), pad=angle_pad)

    axs_phase[0].set_title(r"$(i, j) = (1, 2)$", fontsize=ax_ticks_scale * fontsize, pad=30)
    axs_phase[1].set_title(r"$(i, j) = (1, 3)$", fontsize=ax_ticks_scale * fontsize, pad=30)
    axs_phase[2].set_title(r"$(i, j) = (2, 3)$", fontsize=ax_ticks_scale * fontsize, pad=30)

    rms6_min = np.min(r_total)
    rms6_max = np.max(r_total)
    colormap = 'viridis'
    ax_rms6 = fig.add_subplot(gs[2 * scale + offset:3 * scale - offset_rows, 2 * scale - 20:3 * scale])
    im6 = ax_rms6.pcolormesh(
        np.reshape(np.kron(np.ones(len(delta_one_arr)), delta_one_arr), (len(delta_one_arr), len(delta_one_arr))),
        np.reshape(np.kron(delta_two_arr, np.ones(len(delta_two_arr))), (len(delta_two_arr), len(delta_two_arr))),
        r_total, cmap=colormap, shading='nearest', rasterized=True)
    ax_ins = inset_axes(ax_rms6, width="5%", height="80%", loc='lower right',
                        bbox_to_anchor=(0.1, 0., 1, 1), bbox_transform=ax_rms6.transAxes, borderpad=0)
    cbar_ticks = [rms6_min, 0.5 * (rms6_max + rms6_min), rms6_max]
    cbar = plt.colorbar(im6, cax=ax_ins, orientation='vertical', ticks=cbar_ticks)
    cbar.ax.set_yticklabels([f"{cbar_ticks[i]:.2f}" for i in range(len(cbar_ticks))])
    cbar.ax.tick_params(labelsize=cbar_scale * fontsize)
    ax_ins.set_title(r"$r$", fontsize=cbar_rms_title_scale * fontsize, pad=r_pad)
    ax_rms6.set_xlabel(r"$\Delta_1$", fontsize=ax_label_scale * fontsize)
    ax_rms6.set_ylabel(r"$\Delta_2$", fontsize=ax_label_scale * fontsize)
    ax_rms6.tick_params(axis='x', labelsize=ax_ticks_scale * fontsize)
    ax_rms6.tick_params(axis='y', labelsize=ax_ticks_scale * fontsize)
    if equal_aspect:
        ax_rms6.set_aspect('equal', adjustable='box')
    sums_min = np.min(diff_sums)
    sums_max = np.max(diff_sums)
    colormap = 'magma'
    ax_sum = fig.add_subplot(gs[2 * scale + offset:3 * scale - offset_rows, 0:scale + 20])
    im = ax_sum.pcolormesh(
        np.reshape(np.kron(np.ones(len(delta_one_arr)), delta_one_arr), (len(delta_one_arr), len(delta_one_arr))),
        np.reshape(np.kron(delta_two_arr, np.ones(len(delta_two_arr))), (len(delta_two_arr), len(delta_two_arr))),
        diff_sums, cmap=colormap, shading='nearest', rasterized=True)
    ax_ins = inset_axes(ax_sum, width="5%", height="80%", loc='lower right',
                        # bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = curr_ax.transAxes,
                        bbox_to_anchor=(0.1, 0., 1, 1), bbox_transform=ax_sum.transAxes, borderpad=0)
    cbar_ticks = [sums_min, 0.5 * (sums_max + sums_min), sums_max]
    cbar = plt.colorbar(im, cax=ax_ins, orientation='vertical', ticks=cbar_ticks)
    cbar.ax.set_yticklabels([f"{cbar_ticks[i]:.2f}" for i in range(len(cbar_ticks))])
    cbar.ax.tick_params(labelsize=cbar_scale * fontsize)
    ax_ins.set_title(r"$\Phi_{sum}$", fontsize=cbar_angle_title_scale * fontsize, position=(2.9, 0), pad=angle_pad)
    ax_sum.set_xlabel(r"$\Delta_1$", fontsize=ax_label_scale * fontsize)
    ax_sum.set_ylabel(r"$\Delta_2$", fontsize=ax_label_scale * fontsize)
    ax_sum.tick_params(axis='x', labelsize=ax_ticks_scale * fontsize)
    ax_sum.tick_params(axis='y', labelsize=ax_ticks_scale * fontsize)
    if equal_aspect:
        ax_sum.set_aspect('equal', adjustable='box')
    left_shift = 1.6 / (subfig_label_scale * fontsize)
    up_shift = -0.1 / (subfig_label_scale * fontsize)
    axs_rms[0].text(-2 - subfig_label_scale * fontsize * left_shift, 2 + subfig_label_scale * fontsize * up_shift, "(b)",
                    fontsize=subfig_label_scale * fontsize)
    axs_phase[0].text(-2 - subfig_label_scale * fontsize * left_shift, 2 + subfig_label_scale * fontsize * up_shift, "(a)",
                      fontsize=subfig_label_scale * fontsize)
    ax_rms6.text(-2 - subfig_label_scale * fontsize * left_shift, 2 + subfig_label_scale * fontsize * up_shift, "(d)",
                 fontsize=subfig_label_scale * fontsize)
    ax_sum.text(-2 - subfig_label_scale * fontsize * left_shift, 2 + subfig_label_scale * fontsize * up_shift, "(c)",
                fontsize=subfig_label_scale * fontsize)

    # Save figure
    plt.savefig("rms_phase_plots_text.png", dpi=dpi)