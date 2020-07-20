import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class rungekutta:
    def __init__(self, init_pos, init_vel, init_tau=0.1):

        self.tau = init_tau

        self.dim = init_pos.size
        # safety constants
        self.S1 = 0.9
        self.S2 = 1.3
        self.sigma = 0.001
        # system variables
        self.t = 0
        self.x = init_pos
        self.v = init_vel

        self.eps = []
        self.taus = []
        self.ts = []

    def acc(self, x, v, t):
        return -4 * (np.pi ** 2) / np.sum(x ** 2) * x / np.linalg.norm(x)

    def integrate(self, r_input, v_input, t, tau):
        # calculate K_ and K'_ values (prime denoted as p in code)
        k1 = tau * v_input
        k1p = tau * self.acc(r_input, v_input, t)

        k2 = tau * (v_input + 0.5 * k1p)
        k2p = tau * self.acc(
            r_input + 0.5 * k1, v_input + 0.5 * k1p, t + tau / 2
        )

        k3 = tau * (v_input + 0.5 * k2p)
        k3p = tau * self.acc(
            r_input + 0.5 * k2, v_input + 0.5 * k2p, t + tau / 2
        )

        k4 = tau * (v_input + k3p)
        k4p = tau * self.acc(r_input + k3, v_input + k3p, t + tau)

        r = r_input + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        v = v_input + (k1p + 2 * k2p + 2 * k3p + k4p) / 6

        return r, v

    def adaptive(self):
        while True:  # run until solution is accepted
            # first run
            x1, v1 = self.integrate(self.x, self.v, self.t, self.tau)

            # second run
            xh, vh = self.integrate(self.x, self.v, self.t, self.tau / 2)
            x2, v2 = self.integrate(
                xh, vh, self.t + self.tau / 2, self.tau / 2
            )

            # calculate local error
            epsilon_x = np.linalg.norm(x2 - x1)
            epsilon_v = np.linalg.norm(v2 - v1)
            epsilon = np.sqrt(epsilon_x ** 2 + epsilon_v ** 2)
            self.eps.append(epsilon)
            scale = self.S1 * (self.sigma / epsilon) ** (1 / 5)
            new_tau = self.tau * scale
            if epsilon < self.sigma:  # accept
                self.x = x1
                self.v = v1
                self.t += self.tau
                self.ts.append(self.t)
                self.tau = self.calc_tau(self.tau, new_tau)  # update tau
                self.taus.append(self.tau)
                break
            else:  # reject
                self.tau = self.calc_tau(self.tau, new_tau)

    def calc_tau(self, tau, tau_prime):
        # update tau with given safety parameters
        if tau / self.S2 < self.S1 * tau_prime:
            if self.S1 * tau_prime < self.S2 * tau:
                return self.S1 * tau_prime
            else:
                return self.S1 * tau
        else:
            return tau / self.S2

    def simulate(self, steps):
        trajectory = np.zeros((steps, self.dim))
        for i in tqdm(range(steps)):
            self.adaptive()
            trajectory[i] = self.x
        self.trajectory = trajectory

    def ecc(self):
        return np.sqrt(1 - (self.minor() / self.major()) ** 2)

    def major(self):
        i = np.argmax(self.trajectory[:, 0])
        j = np.argmin(self.trajectory[:, 0])
        return np.linalg.norm(self.trajectory[i] - self.trajectory[j]) / 2

    def minor(self):
        i = np.argmax(self.trajectory[:, 1])
        j = np.argmin(self.trajectory[:, 1])
        return np.linalg.norm(self.trajectory[i] - self.trajectory[j]) / 2

    def T(self):
        i = np.argmin(self.trajectory[:, 0])  # find "most left" point
        j = np.argmax(
            self.trajectory[i:, 0]
        )  # find "most right" point after doing at least half a turn
        return self.ts[i + j]

    def plot(self):
        plt.title("Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 0.5)

        plt.scatter(self.trajectory[:, 0], self.trajectory[:, 1], marker=".")
        plt.scatter(0, 0, marker="x")

        # for i, txt in enumerate(self.ts):
        #     plt.annotate(txt, (self.trajectory[i, 0], self.trajectory[i, 1]))
        plt.show()
        plt.close()


T = []
a = []

pos0 = np.array([1, 0])
vel0 = np.array([0, np.pi / 2])

simulator = rungekutta(pos0, vel0)
simulator.simulate(50)
simulator.plot()
T.append(simulator.T())
a.append(simulator.major())

vel0 = np.array([0, np.pi])

simulator = rungekutta(pos0, vel0)
simulator.simulate(50)
simulator.plot()
T.append(simulator.T())
a.append(simulator.major())

vel0 = np.array([0, (3 / 4) * np.pi])

simulator = rungekutta(pos0, vel0)
simulator.simulate(50)
simulator.plot()
T.append(simulator.T())
a.append(simulator.major())

for i in tqdm(range(10)):
    vel0 = np.array([0, np.pi / 2 + i * np.pi / 20])

    simulator = rungekutta(pos0, vel0)
    simulator.simulate(50)
    T.append(simulator.T())
    a.append(simulator.major())

T = np.array(T)
print(T)
a = np.array(a)
plt.scatter(T ** 2, a ** 3, label="Simulation")
plt.plot(
    [0.9 * np.min(T) ** 2, 1.1 * np.max(T) ** 2],
    [0.9 * np.min(T) ** 2, 1.1 * np.max(T) ** 2],
    label="Theory",
)
plt.legend()
plt.ylabel("$a^3$")
plt.xlabel("$T^2$")
plt.show()
