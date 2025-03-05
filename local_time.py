import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors


def fixed_point_iteration(x: np.ndarray, feedback: float = 1.0):
    if feedback <= .0:
        return 0.

    n = x.size
    x_sorted = np.sort(x)/feedback
    idx = x_sorted >= feedback*(np.arange(1, n + 1)*x_sorted - x_sorted.cumsum())/n
    idx = np.where(idx == True)[0]

    if idx.size > 0:
        idx = idx[0]
        return -feedback*x_sorted[:idx].sum()/(n - feedback*idx)
    else:
        idx = math.ceil(1/feedback)
        return None

def particle_system(initial_condition: np.ndarray, time_step: float, steps = int, feedback: float = 1.0, volatility: float = 1.0):
    size = initial_condition.size
    solution = np.zeros((steps + 1, size))
    atom = np.zeros(steps + 1)
    local_time = np.zeros((steps + 1, size))

    solution[0, :] = initial_condition

    blow_up = steps

    for t in range(0, steps):
        delta_w = np.random.normal(0., time_step**(1/2), size)
        x = solution[t, :] + volatility*delta_w
        atom[t] = (np.sort(x, -1).cumsum(-1) < 0.).mean(-1)

        if x.min() >= 0.:
            delta_l = 0.
        else:
            delta_l = fixed_point_iteration(x, feedback=feedback)

        if delta_l is None:
            blow_up = t
            break
        
        delta_local_time = -np.minimum(x - feedback*delta_l, 0.)
        solution[t + 1, :] = x - feedback*delta_l + delta_local_time
        local_time[t + 1, :] = local_time[t, :] + delta_local_time

        if t % 10**4 == 0:
            print(f"Current step: {t}")

    return solution, local_time, atom, blow_up


def particle_system_no_history(initial_condition: np.ndarray, time_step: float, steps = int, feedback: float = 1.0, volatility: float = 1.0):
    size = initial_condition.size
    solution = initial_condition

    for t in range(0, steps):
        delta_w = np.random.normal(0., time_step**(1/2), size)
        x = solution + volatility*delta_w

        if x.min() >= 0.:
            delta_l = 0.
        else:
            delta_l = fixed_point_iteration(x, feedback=feedback)

        if delta_l is None:
            break
        
        delta_local_time = -np.minimum(x - feedback*delta_l, 0.)
        solution = x - feedback*delta_l + delta_local_time

        if t % 10**4 == 0:
            print(f"Current step: {t}")

    return solution

def self_similar_profile(x: np.array, feedback: float, normalisation: float = 1.0):
        return normalisation**2/(1 - feedback)*np.exp(-normalisation**2*(feedback*x/(1 - feedback) + x**2/2))

def solve_normalisation(feedback: float, steps: int, initial_condition: float = math.sqrt(2/math.pi)):
    constant = initial_condition
    step_size = feedback/steps
    t = 0.
    
    for _ in range(steps):
        constant = constant + step_size*constant*(constant**2 - (1 - t))/((1 - t)*(-constant**2*t + (1 - t)))
        t += step_size
    
    return constant

def density_video(solution: np.array, times: np.array, feedback: float = 1.0):

    fig, ax = plt.subplots()
        
    if feedback < 1.:
        samples = solution/solution.mean()
        normalisation = solve_normalisation(feedback, 10**5)
        x = np.linspace(min(samples), max(samples), solution.shape[1])
        true_density = self_similar_profile(x, feedback=feedback, normalisation=normalisation)
        param = normalisation/(1 - feedback)
        label = "Self-Similar Profile"
        prefix = "Rescaled "
    elif feedback == 1.:
        samples = solution
        param = (samples[0, :].mean())**(-1)
        x = np.linspace(samples.min(), samples.max(), solution.shape[1])
        true_density = param*np.exp(-param*x)
        label = "Stationary Profile"
        prefix = ""
    else:
        print("No stationary behaviour")

    hist_values, bin_edges = np.histogram(samples[0, :], bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    orange = default_colors[1]

    ax.bar(bin_centers, hist_values, width=bin_edges[1] - bin_edges[0], alpha=0.6, label=prefix + "Empirical Density")
    ax.plot(x, true_density, label=label, c=orange)
    ax.set_ylim(0., 1.5*param)
    ax.set_ylabel('Density')
    ax.set_xlabel(r'$x$')
    ax.legend()
    

    def update(frame):
        time = times[frame]
        print(time)

        ax.clear()

        hist_values, bin_edges = np.histogram(samples[frame, :], bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(bin_centers, hist_values, width=bin_edges[1] - bin_edges[0], alpha=0.6, label=prefix + "Empirical Density")
        ax.plot(x, true_density, label=label, c=orange)
        ax.set_ylim(0., 1.5*param)
    
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$x$')
        ax.legend()

        ax.set_title(label + f' at Time {time:.1f}')

        return ax,

    ani = FuncAnimation(fig, update, frames=times.size, blit=False)

    # Save the animation as a video
    ani.save('data/local_time/stationary_video.mp4', fps=100, dpi=300)
    print("test")

    return fig, ax

def reflected_bm(initial_condition: float, time_step: float, steps = int):
    rbm = np.zeros(steps + 1)
    rbm[0] = initial_condition
    rbm[1:] = np.abs(initial_condition + np.random.normal(0., time_step**(1/2), steps).cumsum())

    return rbm


if __name__ == "__main__":

    # Multiple feedbacks

    np.random.seed(1)

    initial_condition = 0.5*np.ones(10**4)
    steps = 10**4
    final_time = 1.
    time_step = final_time/steps
    final_time = time_step*steps
    feedbacks = np.linspace(0., 2., 9)
    solutions = np.zeros((feedbacks.size, steps + 1, initial_condition.size))
    local_times = np.zeros((feedbacks.size, steps + 1))
    blow_ups = np.zeros(feedbacks.size, dtype=int)
    volatility = 1.0

    generate = False
    if generate:
        for i in range(feedbacks.size):
            print(f"Current step: {i}")
            np.random.seed(1)
            solution, local_time, _, blow_up = particle_system(initial_condition, time_step, steps, feedback=feedbacks[i], volatility=volatility)
            solutions[i, ...] = solution
            local_times[i, ...] = local_time.mean(-1)
            blow_ups[i] = blow_up

            # Save data in folder data/local_time
            np.savez('data/local_time/multiple_feedbacks', solutions=solutions, local_times=local_times, blow_ups=blow_ups, step=i)

    # Load data from folder data/local_time
    data = np.load('data/local_time/multiple_feedbacks.npz')
    solutions = data['solutions']
    local_times = data['local_times']
    blow_ups = data['blow_ups']

    fig, ax = plt.subplots()

    times = np.linspace(0., final_time, steps + 1)
    colormap = cm.get_cmap('Blues', feedbacks.size - 1)

    def iterated_logarithm(t: float, speed: float = 1., constant: float = 1.):
        return  np.log(np.log(t + 1) + 1)/5
    for i in range(feedbacks.size):
        color = colormap(i)
        ax.plot(times[:blow_ups[i] + 1], local_times[i, :blow_ups[i] + 1], color=color)

    
    norm = mcolors.Normalize(vmin=0., vmax=feedbacks[-1])
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # Add colorbar as a legend
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'$\alpha$')

    ax.set_xlim(0., final_time)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\ell_t$')
    plt.show()


    # Self-similar and stationary solutions

    np.random.seed(1)

    n = 10**5
    initial_condition = 0.5*np.ones(n)
    steps = 10**5
    final_time = 10.
    time_step = final_time/steps
    final_time = time_step*steps
    feedback = .5
    volatility = 1.

    generate = False
    if generate:
        solution = particle_system_no_history(initial_condition, time_step, steps, feedback, volatility)

        if feedback < 1.:
            # Save data in folder data/local_time
            np.savez('data/local_time/self_similar', solution=solution)
        else:
            # Save data in folder data/local_time
            np.savez('data/local_time/stationary', solution=solution)

    # Load data from folder data/local_time
    if feedback < 1.:
        data = np.load('data/local_time/self_similar.npz')
    else:
        data = np.load('data/local_time/stationary.npz')
    solution = data['solution']

    if feedback < 1.:
        samples = solution/solution.mean()
    else:
        samples = solution
    hist_values, bin_edges = np.histogram(samples, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    normalisation = solve_normalisation(feedback, 10**5)
        
    x = np.linspace(min(samples), max(samples), n)
    if feedback < 1.:
        true_density = self_similar_profile(x, feedback=feedback, normalisation=normalisation)
    elif feedback == 1.:
        true_density = np.exp(-x/initial_condition.mean())/initial_condition.mean()
    else:
        print("No stationary behaviour")

    # Plot both the empirical and true densities
    fig, ax = plt.subplots()
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    orange = default_colors[1]
    if feedback < 1.:
        label = "Self-Similar Profile"
        prefix = "Rescaled "
    else:
        label = "Stationary Profile"
        prefix = ""
    ax.bar(bin_centers, hist_values, width=bin_edges[1] - bin_edges[0], alpha=0.6, label=prefix + "Empirical Density")
    ax.plot(x, true_density, label=label, c=orange)
    
    ax.set_ylabel('Density')
    ax.set_xlabel(r'$x$')
    ax.legend()
    plt.show()


    # Self-similar and stationary solutions video

    np.random.seed(1)

    n = 10**4
    initial_condition = 0.5*np.ones(n)
    steps = 10**4
    final_time = 10.
    time_step = final_time/steps
    final_time = time_step*steps
    feedback = 1.
    volatility = 1.

    generate = False
    if generate:
        solution, local_time, _, _ = particle_system(initial_condition, time_step, steps, feedback, volatility)
        print('Data generated')

        if feedback < 1.:
            # Save data in folder data/local_time
            np.savez('data/local_time/self_similar_video', solution=solution)
        else:
            # Save data in folder data/local_time
            np.savez('data/local_time/stationary_video', solution=solution)

    # Load data from folder data/local_time
    if feedback < 1.:
        data = np.load('data/local_time/self_similar_video.npz')
    else:
        data = np.load('data/local_time/stationary_video.npz')
    solution = data['solution']
    print('Data loaded')

    density_video(solution, np.linspace(0., final_time, steps + 1), feedback=feedback)


    # Atom

    np.random.seed(1)

    initial_condition = 0.5*np.ones(10**4)
    steps = 10**5
    final_time = 1.
    time_step = final_time/steps
    final_time = time_step*steps
    feedback = 1.5
    volatility = 1.

    generate = False
    if generate:
        solution, local_time, atom, blow_up = particle_system(initial_condition, time_step, steps, feedback, volatility)
        local_time_mean = local_time.mean(-1)
    
        # Save data in folder data/local_time
        np.savez('data/local_time/single_feedback', solution=solution, local_time=local_time, atom=atom, blow_up=blow_up)

    # Load data from folder data/local_time
    data = np.load('data/local_time/single_feedback.npz')
    solution = data['solution']
    local_time = data['local_time']
    atom = data['atom']
    blow_up = data['blow_up']
    local_time_mean = local_time.mean(-1)

    times = np.linspace(0., final_time, steps + 1)

    fig, ax = plt.subplots()
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    red = default_colors[3]

    ax.vlines(times[:blow_up + 1], ymin=0, ymax=atom[:blow_up + 1], linewidth=0.5)
    ax.axhline(1/feedback, c=red, linestyle='--')

    yticks = ax.get_yticks()[1:-1]
    ax.set_yticks(list(yticks) + [1/feedback])

    yticklabels = [str(round(tick, 1)) for tick in yticks]
    yticklabels.append(r'$1/\alpha$')
    ax.set_yticklabels(yticklabels)

    ax.set_xlim(0., final_time)
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\mathbb{P}(X_t = 0)$')
    plt.show()


    # Reflected Brownian motion

    np.random.seed(2)

    initial_condition = 0.3
    steps = 10**5
    final_time = 1.
    time_step = final_time/steps
    final_time = time_step*steps
    rbm = reflected_bm(initial_condition, time_step, steps)

    plt.plot(np.linspace(0., final_time, steps + 1), rbm, linewidth=0.75)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$X_t$')
    plt.ylim(0., 1.1*rbm.max())
    plt.show()
