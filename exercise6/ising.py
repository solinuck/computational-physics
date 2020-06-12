import numpy as np
from pathlib import Path
import itertools

from logs import Logging


def init_configuration(n):
    return np.random.choice([-1, 1], n)


def calc_energy(spins):
    return -np.sum(spins[:-1] * spins[1:])


def mmc(n_spins, beta):
    spins = init_configuration(n_spins)
    current_energy = calc_energy(spins)

    for i in range(n_spins):  # range(n_spins)

        j = np.random.randint(n_spins)
        # change spin
        spins[j] *= -1

        flip_energy = calc_energy(spins)

        delta_e = flip_energy - current_energy

        q = np.exp(-beta * delta_e)

        r = np.random.random()

        if q > r:
            current_energy = flip_energy
        else:
            # set spin back
            spins[j] *= -1
    return current_energy


def calc_average(n_samples, energies):
    return 1 / n_samples * np.sum(energies)


random_seed = 1
np.random.seed(random_seed)
ts = np.arange(0.2, 4.2, 0.2)
n_samples = [1000, 10000]
n_spins = [10, 100]
mode = "1D"

logs = Path("logs")

dirs = [
    logs.joinpath(mode, f"n_{n_spin}", f"ns_{n_sample}")
    for (n_spin, n_sample) in itertools.product(n_spins, n_samples)
]

for dir in dirs:
    dir.mkdir(parents=True, exist_ok=True)


loggers = {
    f"{dir.parts[2]}_{dir.parts[3]}": Logging(
        f"{dir.parts[2]}_{dir.parts[3]}", dir.joinpath("file")
    )
    for dir in dirs
}


for (n_spin, n_sample) in itertools.product(n_spins, n_samples):
    log_name = f"n_{n_spin}_ns_{n_sample}"
    loggers[log_name].logger.info(f"n_spins = {n_spin}, n_samples = {n_sample}")
    loggers[log_name].format_log("T", "beta", "U_MC", "C_MC", "U_THEO", "C_THEO", "acc")
    for t in ts:
        beta = 1 / t

        energies = np.array([mmc(n_spin, beta) for n in range(n_sample)])

        u_theory = -(n_spin - 1) / n_spin * np.tanh(beta)
        c_theory = (n_spin - 1) / n_spin * (beta / np.cosh(beta)) ** 2

        u_mc = calc_average(n_sample, energies) / n_spin
        c_mc = beta ** 2 * (calc_average(n_sample, energies ** 2) / n_spin - u_mc ** 2)

        u_acc = np.abs(u_theory - u_mc) / np.abs(u_theory)

        loggers[log_name].format_log(
            np.round(t, 2), u_mc, c_mc, u_theory, c_theory, u_acc, format_nums=True
        )
    loggers[log_name].logger.info("")
