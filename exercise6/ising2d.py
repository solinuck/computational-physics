import numpy as np
from pathlib import Path
import itertools

from logs import Logging


def init_configuration2d(n):
    return np.random.choice([-1, 1], (n, n))


def calc_energy2d(spins):
    return -np.sum(spins[:-1, :] * spins[1:, :]) - np.sum(spins[:, :-1] * spins[:, 1:])


def mmc2(n_spins, beta):
    spins = init_configuration2d(n_spins)
    current_energy = calc_energy2d(spins)

    for n in range(n_spins ** 2):  # range(n_spins)

        i = np.random.randint(n_spins)
        j = np.random.randint(n_spins)
        # change spin
        spins[i, j] *= -1

        flip_energy = calc_energy2d(spins)

        delta_e = flip_energy - current_energy

        q = np.exp(-beta * delta_e)

        r = np.random.random()

        if q > r:
            current_energy = flip_energy
        else:
            # set spin back
            spins[i, j] *= -1
    return current_energy, np.sum(spins)


def calc_average(n_samples, f):
    return 1 / n_samples * np.sum(f)


random_seed = 1
np.random.seed(random_seed)
ts = np.arange(0.2, 4.2, 0.2)
n_samples = [1000, 10000]
n_spins = [10, 50, 100]
mode = "2D"


logs = Path("logs_")

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
    loggers[log_name].format_log(
        "T", "beta", "U_MC", "C_MC", "M_MC", "U_THEO", "C_THEO", "M_THEO", "acc"
    )

    tc = 2 / (np.log(1 + 2 ** 0.5))

    for t in ts:
        beta = 1 / t

        result = np.array([(mmc2(n_spin, beta)) for n in range(n_sample)])

        energies = result[:, 0]
        spin_sums = result[:, 1]
        from IPython import embed

        embed()
        u_theory = -(n_spin ** 2 - 1) / n_spin ** 2 * np.tanh(beta)
        c_theory = (n_spin ** 2 - 1) / n_spin ** 2 * (beta / np.cosh(beta)) ** 2
        if t < tc:
            m_theory = (1 - np.sinh(2 * beta) ** -4) ** (1 / 8)
        else:
            m_theory = 0

        u_mcN = calc_average(n_sample, energies)
        c_mcN = beta ** 2 * (calc_average(n_sample, energies ** 2) - u_mcN ** 2)
        m_mc = calc_average(n_sample, spin_sums) ** 2 / n_spin ** 2

        u_mc = u_mcN / n_spin ** 2
        c_mc = c_mcN / n_spin ** 2

        u_acc = np.abs(u_theory - u_mc) / np.abs(u_theory)

        loggers[log_name].format_log(
            np.round(t, 2),
            beta,
            u_mc,
            c_mc,
            m_mc,
            u_theory,
            c_theory,
            m_theory,
            u_acc,
            format_nums=True,
        )
    loggers[log_name].logger.info("")
