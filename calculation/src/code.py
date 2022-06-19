import sys
import time
import itertools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


class DiffusionDiffEquationSolver:
    emitters = []
    area = {}
    history = {}
    current_time = 0
    lattice = None
    
    def __init__(self, k_1, k_2, l,
                 n, x, y, t_step,
                 u_crit,
                 emitters, phi,
                 initial):
        self.k_1 = k_1
        self.k_2 = k_2
        self.l = l
        self.n = n

        self.area['x'] = x
        self.area['y'] = y
        self.nodesx = np.linspace(self.area['x'][0], self.area['x'][1], n)
        self.nodesy = np.linspace(self.area['y'][0], self.area['y'][1], n)
        
        self.nodes = np.array(list(itertools.product([i for i in range(1, n-1)], repeat=2)))

        self.x_step = self.nodesx[1] - self.nodesx[0]
        self.y_step = self.nodesy[1] - self.nodesy[0]
        self.t_step = t_step
        if np.power(self.x_step, 2) * 0.4 < self.t_step:
            print('The paramethers lead to unstable model')
            self.t_step = np.power(self.x_step , 2)

        self.u_crit = u_crit
        self.phi = phi
        self.emitters = emitters
        # unitialize lattice
        self.lattice = np.zeros((n, n))
        for i in range(self.lattice.shape[0]):
            for j in range(self.lattice.shape[1]):
                self.lattice[i, j] = initial(i, j)

    def _build_discrete_eq_coefs_next(self, i, j):
        coefs = np.array([
            [- self.k_1 * self.l / np.power(self.x_step, 2)],
            [- self.k_2 * self.l / np.power(self.y_step, 2)],
            [1/self.t_step +
             2 * self.k_1 * (1-self.l)/np.power(self.x_step, 2) -
             2 * self.k_2 * (1-self.l)/np.power(self.y_step, 2)],
            [- self.k_1 * self.l / np.power(self.x_step, 2)],
            [- self.k_2 * self.l / np.power(self.y_step, 2)],
        ])
        
        clearance_strength = 1
        if self.lattice[i, j] >= self.u_crit:
            coefs[0, 0] += self.l * clearance_strength
            coefs[1, 0] += self.l * clearance_strength
            coefs[2, 0] += (1 - 2 * self.l) * clearance_strength
        return coefs
    
    def _build_discrete_coefs_bias(self, i, j):
        coefs = np.array([
            [self.k_1 * (1 - self.l) / np.power(self.x_step, 2)],
            [self.k_2 * (1 - self.l) / np.power(self.y_step, 2)],
            [1/self.t_step -
             2 * self.k_1 * self.l / np.power(self.x_step, 2) +
             2 * self.k_2 * self.l / np.power(self.y_step, 2)],
            [self.k_1 * (1 - self.l) / np.power(self.x_step, 2)],
            [self.k_2 * (1 - self.l) / np.power(self.y_step, 2)],
        ])
        clearance_strength = 1
        clearance_addition = 0
        if self.lattice[i, j] >= self.u_crit:
            coefs[0, 0] += (- 1 + self.l) * clearance_strength + clearance_addition
            coefs[1, 0] += (- 1 + self.l) * clearance_strength + clearance_addition
            coefs[2, 0] += (3 - 2 * self.l) * clearance_strength + clearance_addition
        return coefs

    def _emition_at(self, i, j):
        if (i, j) in self.emitters:
            return self.phi
        return 0

    def _compute_bias_at(self, i, j):
        coefs = self._build_discrete_coefs_bias(i, j)
        concentration = np.array([
            [self.lattice[i+1, j]],
            [self.lattice[i, j+1]],
            [self.lattice[i, j]],
            [self.lattice[i-1, j]],
            [self.lattice[i, j-1]],
        ])
        cleaning = 0
        if self.lattice[i, j] >= self.u_crit:
            cleaning = 100
        return np.sum(coefs * concentration) + self._emition_at(i, j) - cleaning
    
    def _build_next_system(self):        
        n = np.power(self.n - 2, 2)
        next_lattice_coefs = np.zeros((n, n))

        def commit_to_next_lattice(eqn, i, j, value):
            variable_n = (self.n - 2) * (i - 1) + (j - 1)
            if i == 0 or j == 0 or i == self.n-1 or j == self.n-1:
                return
            next_lattice_coefs[eqn, variable_n] = value

        bias = np.zeros((n, 1))
        for eqn, node_idx in enumerate(self.nodes):
            b = self._compute_bias_at(node_idx[0], node_idx[1])
            row_idx = (self.n - 2) * (node_idx[0] - 1) + (node_idx[1] - 1)
            bias[row_idx, 0] = b
            left_coefs = self._build_discrete_eq_coefs_next(node_idx[0], node_idx[1])

            commit_to_next_lattice(eqn, node_idx[0]+1, node_idx[1], left_coefs[0, 0])
            commit_to_next_lattice(eqn, node_idx[0], node_idx[1]+1, left_coefs[1, 0])
            commit_to_next_lattice(eqn, node_idx[0], node_idx[1], left_coefs[2, 0])
            commit_to_next_lattice(eqn, node_idx[0]-1, node_idx[1], left_coefs[3, 0])
            commit_to_next_lattice(eqn, node_idx[0], node_idx[1]-1, left_coefs[4, 0])
        return next_lattice_coefs, bias
    
    def step(self):
        lattice_coefs, bias = self._build_next_system()
        # print(f"lattice coefs = {lattice_coefs}")
        # print(f"bias = {bias}")
        solution = np.linalg.solve(lattice_coefs, bias)
        # print(f"solution is = {solution}")
        self.history[self.current_time] = np.copy(self.lattice)
        self.current_time += 1
        for node_idx, node in enumerate(self.nodes):
            node_value = solution[node_idx, 0]
            self.lattice[node[0], node[1]] = node_value if node_value >= 0 else 0
        return self.lattice
        

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(f"Usage: {sys.argv[0]} <k_1> <k_2> <lambda> <emitter-strength> <emitter1-x> <emitter1-y> <emitter2-x> <emitter2-y> <size> <frames>")
        exit()
    _, k_1, k_2, l, emitter_strength, emitter1x, emitter1y, emitter2x, emitter2y, n, frames = sys.argv

    k_1 = float(k_1)
    k_2 = float(k_2)
    l = float(l)
    phi = float(emitter_strength)
    emitter1x = int(emitter1x)
    emitter1y = int(emitter1y)
    emitter2x = int(emitter2x)
    emitter2y = int(emitter2y)
    n = int(n)
    frames = int(frames)

    # k_1, k_2 = [0.5, 0.3]
    # k_1, k_2 = [0.5, 0.5]
    # l = 0.4                      # lambda
    # n = 15                        # number of nodes

    x, y = [(0, 10), (0, 10)]      # observed area
    t_step = 1                     # time step
    u_crit = 100                   # critical concentration
    # phi = 5                      # emitters strength
    emitters = [(emitter1x, emitter1y), (emitter2x, emitter2y)]  # emitters location at node
    initial = lambda x, y: 0

    solver = DiffusionDiffEquationSolver(k_1, k_2, l,
                                         n, x, y,
                                         t_step, u_crit,
                                         emitters, phi,
                                         initial)
    
    frames_n = frames
    for i in range(frames_n):
        solver.step()

    print(f"Solved system for {frames_n} steps")
    print("System data:")
    print(f"    emitters  = {emitters}")
    print(f"    time step = {solver.t_step}")
    print(f"    x step    = {solver.x_step}")
    print(f"    y step    = {solver.y_step}")
    print(f"    k_1  = {solver.k_1}")
    print(f"    k_2  = {solver.k_2}")
    print(f"    l    = {solver.l}")
    print(f"    n    = {solver.n}")

    max_concentration = np.max(np.array([solver.history[i] for i in solver.history]))

    animate_method = 'all'

    def animate_surface():
        plt.rcParams["figure.figsize"] = [7.50, 7.50]
        plt.rcParams["figure.autolayout"] = True

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.view_init(60, 35)
        ax.view_init(30, 0)
        title = ax.set_title('Diffusion Process')
        data = solver.history[0]
        X, Y = np.meshgrid(solver.nodesx, solver.nodesy)
        # ax.set_zlim(-0.1, u_crit * 1.1)
        ax.set_zlim(-0.1, max_concentration)
        ax.plot_surface(X, Y, data,
                        cmap='Reds', edgecolor='black')

        def update(i):
            print(f"\r  rendering frame {i}", end='')
            ax.view_init(elev=30., azim=i + 5)
            data = solver.history[i]
            ax.plot_surface(X, Y, data,
                            cmap='Reds', edgecolor='black')
            title.set_text(f"Diffusion process, time index = {i} with {k_1=}, {k_2=}, {l=}")

        ani = animation.FuncAnimation(fig, update, frames_n,
                                      interval=200)
        ani.save(f"surface-animation-{n}-{int(time.time())}.mp4")

    def animate_heatmap():
        plt.rcParams["figure.figsize"] = [7.50, 7.50]
        plt.rcParams["figure.autolayout"] = True

        fig = plt.figure()
        kwargs = {
            'square': True,
            'vmin': 0,
            'cmap': 'hot',
            'robust': True,
            # 'annot': True,
        }

        def init():
            sns.heatmap(solver.history[0], cbar=True, **kwargs)

        def animate(i):
            print(f"\r  rendering frame {i}", end='')
            data = solver.history[i]
            sns.heatmap(data, cbar=False, **kwargs)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=frames_n, repeat=False)
        anim.save(f"heatmap-animation-{n}-{int(time.time())}.mp4")

    def animate_all():
        plt.rcParams["figure.figsize"] = [15.0, 7.50]
        plt.rcParams["figure.autolayout"] = True
        # fig, (ax_surf, ax_heat) = plt.subplots(1, 2)

        fig = plt.figure()
        ax_surf = fig.add_subplot(1, 2, 1, projection=Axes3D.name)
        ax_surf.view_init(30, 0)
        ax_heat = fig.add_subplot(1, 2, 2)

        title = fig.suptitle('Diffusion process')
        ax_surf.view_init(30, 0)

        data = solver.history[0]
        X, Y = np.meshgrid(solver.nodesx, solver.nodesy)
        # ax_surf.set_zlim(-0.1, max_concentration)
        ax_surf.plot_surface(X, Y, data,
                             cmap='Reds', edgecolor='black')

        heat_kwargs = {
            # 'square': True,
            'vmin': 0,
            # 'vmax': max_concentration,
            'cmap': 'hot',
            # 'robust': True,
            # 'annot': True,
        }

        def init():
            # sns.heatmap(solver.history[0], cbar=True, **heat_kwargs)
            ax_heat.pcolormesh(X, Y, solver.history[0], **heat_kwargs)
            # ax_heat.colorbar()

        def animate(i):
            print(f"\r  rendering frame {i}", end='')
            data = solver.history[i]

            ax_heat.pcolormesh(X, Y, data, **heat_kwargs)
            # ax_heat.colorbar()

            ax_surf.view_init(elev=30., azim=i + 5)
            ax_surf.plot_surface(X, Y, data,
                                 cmap='Reds', edgecolor='black')
            title.set_text(f"Diffusion process, time index = {i} with {k_1=}, {k_2=}, {l=}")

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=frames_n, repeat=False)
        anim.save(f"animation-{n}-{int(time.time())}.mp4")

    def plot_all(moment):
        plt.rcParams["figure.figsize"] = [15.0, 7.50]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax_surf = fig.add_subplot(1, 2, 1, projection=Axes3D.name)
        ax_surf.view_init(30, 0)
        ax_heat = fig.add_subplot(1, 2, 2)

        heat_kwargs = {
            'vmin': 0,
            # 'vmax': max_concentration,
            'cmap': 'hot',
        }

        title = fig.suptitle('Diffusion process')
        title.set_text(f"Diffusion process, time index = {i} with {k_1=}, {k_2=}, {l=}")
        ax_surf.view_init(30, 0)

        data = solver.history[moment]
        X, Y = np.meshgrid(solver.nodesx, solver.nodesy)
        # ax_surf.set_zlim(-0.1, max_concentration)
        ax_surf.plot_surface(X, Y, data,
                             cmap='Reds', edgecolor='black')

        ax_heat.pcolormesh(X, Y, data, **heat_kwargs)
        fig.savefig(f"plot({moment})-{n}-{int(time.time())}.png")

    plot_all(int(frames_n * 0.95))

    if animate_method == 'heatmap':
        print('Heat animation process')
        animate_heatmap()
    if animate_method == 'surface':
        print('Surface animation process')
        animate_surface()
    if animate_method == 'all':
        print('Surface and heatmap animation process')
        animate_all()
