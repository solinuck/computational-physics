import numpy as np
from pathlib import Path
import itertools

from logs import Logging


def init_configuration(n):
    return np.random.choice([-1, 1], n)


def calc_energy(spins):
    return -np.sum(spins[1:] * spins[:-1])


def mmc(n_samples, n_spins, beta):
    spins = init_configuration(n_spins)
    current_energy = calc_energy(spins)

    energies = np.empty(n_samples)

    for i in range(n_samples):
        j = np.random.randint(n_spins)
        # change spin
        spins[j] *= -1

        flip_energy = calc_energy(spins)

        energy_change = flip_energy - current_energy

        q = np.exp(-beta * energy_change)

        r = np.random.random()

        if q > r:
            current_energy = flip_energy
        else:
            # set spin back
            spins[j] *= -1

        energies[i] = current_energy
    return energies


def calc_average(n_samples, energies):
    return 1 / n_samples * np.sum(energies)


random_seed = 1
np.random.seed(random_seed)
ts = np.arange(0.2, 4.2, 0.2)
n_samples = [1000, 10000]
n_spins = [10, 100]

logs = Path("logs")

dirs = [
    logs.joinpath(f"n_{n_spin}", f"ns_{n_sample}")
    for (n_spin, n_sample) in itertools.product(n_spins, n_samples)
]

for dir in dirs:
    dir.mkdir(parents=True, exist_ok=True)

loggers = {
    f"{dir.parts[1]}_{dir.parts[2]}": Logging(
        f"{dir.parts[1]}_{dir.parts[2]}", dir.joinpath("file")
    )
    for dir in dirs
}


for (n_spin, n_sample) in itertools.product(n_spins, n_samples):
    loggers[f"n_{n_spin}_ns_{n_sample}"].logger.info(
        f"n_spins = {n_spin}, n_samples = {n_sample}"
    )
    loggers[f"n_{n_spin}_ns_{n_sample}"].format_log(
        "T", "beta", "U_MC", "C_MC", "U_THEO", "C_THEO", "acc"
    )
    for t in ts:
        beta = 1 / t
        energies = mmc(n_sample, n_spin, beta)

        u_theory = -(n_spin - 1) / n_spin * np.tanh(beta)
        c_theory = (n_spin - 1) / n_spin * (beta / np.cosh(beta)) ** 2

        u_mc = calc_average(n_sample, energies) / n_spin
        c_mc = beta ** 2 * (calc_average(n_sample, energies ** 2) - u_mc ** 2) / n_spin

        u_acc = np.abs(u_theory - u_mc) / np.abs(u_theory)

        loggers[f"n_{n_spin}_ns_{n_sample}"].format_log(
            np.round(t, 2),
            beta,
            u_mc,
            c_mc,
            u_theory,
            c_theory,
            u_acc,
            format_nums=True,
        )
    loggers[f"n_{n_spin}_ns_{n_sample}"].logger.info("")
