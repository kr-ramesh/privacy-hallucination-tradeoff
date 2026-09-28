import numpy as np
from dp_accounting import dp_event
from dp_accounting.pld import accountant, common
from dp_accounting.rdp import rdp_privacy_accountant
from prv_accountant import Accountant
from scipy import optimize


def find_noise_multiplier(sampling_probability, num_steps, target_epsilon, target_delta, eps_error=0.1):
    def compute_epsilon(mu):
        return Accountant(noise_multiplier=mu, sampling_probability=sampling_probability, delta=target_delta,
                          max_compositions=num_steps, eps_error=eps_error / 2).compute_epsilon(num_steps)

    mu_max, mu_R, eps_R = 100.0, 1.0, float("inf")
    while eps_R > target_epsilon:
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)[2]
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError("Finding a suitable noise multiplier has not converged. "
                               "Try increasing target epsilon or decreasing sampling probability.")
    mu_L, eps_L = mu_R, eps_R
    while eps_L < target_epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)[0]
    has_converged, bracket = False, [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1] - bracket[0]) * 0.01
        mu_guess = optimize.root_scalar(lambda mu: compute_epsilon(mu)[2] - target_epsilon, bracket=bracket, xtol=mu_err).root
        bracket = [mu_guess - mu_err, mu_guess + mu_err]
        has_converged = (compute_epsilon(mu_guess - mu_err)[2] - compute_epsilon(mu_guess + mu_err)[0]) < 2 * eps_error
    assert compute_epsilon(bracket[1])[2] < target_epsilon + eps_error
    return bracket[1]


def compute_noise_multiplier_with_pld(*, num_examples, batch_size, epochs, delta, target_epsilon=16.0, max_grad_norm=1.0):
    steps = int((num_examples * epochs) // batch_size)
    sampling_probability = batch_size / num_examples
    print(f"Computing noise multiplier for num_examples={num_examples}, batch_size={batch_size}, epochs={epochs}, delta={delta}, target_epsilon={target_epsilon}")
    print(f"Sampling probability: {sampling_probability}, Steps: {steps}")
    all_eps_std_dev = accountant.get_smallest_subsampled_gaussian_noise(
        privacy_parameters=common.DifferentialPrivacyParameters(target_epsilon, delta),
        num_queries=steps, sensitivity=max_grad_norm, sampling_prob=sampling_probability)
    opacus_nm = find_noise_multiplier(sampling_probability=sampling_probability, num_steps=steps, target_epsilon=target_epsilon, target_delta=delta)
    print(f"Using Opacus-based accountant, noise_multiplier={opacus_nm * max_grad_norm:.3f} for delta={delta:.1e} for epsilon={target_epsilon:.3f}")
    print(f"Using PLD accountant, stddev={all_eps_std_dev:.3f}, stddev normalized={all_eps_std_dev / max_grad_norm:.3f} for delta={delta:.1e} for epsilon={target_epsilon:.3f}")
    return all_eps_std_dev


def compute_privacy_pld_accountant(*, num_examples, batch_size, epochs, noise_multiplier, delta):
    steps = int((num_examples * epochs) // batch_size)
    sampling_probability = batch_size / num_examples
    rdp_acc = rdp_privacy_accountant.RdpAccountant()
    for _ in range(steps):
        rdp_acc.compose(dp_event.PoissonSampledDpEvent(sampling_probability, dp_event.GaussianDpEvent(noise_multiplier)))
    epsilon_rdp = rdp_acc.get_epsilon(target_delta=delta)
    print(f"Using RDP accountant, epsilon={epsilon_rdp:.3f} for delta={delta:.1e}")
    return epsilon_rdp
