import mlrose_hiive
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator
import matplotlib.pyplot as plt
import time


def RHC(problem_no_fit, random_state):
    print("using RHC algorithm")
    best_state_HC, best_fitness_HC, fitness_curve_HC = mlrose_hiive.random_hill_climb(problem_no_fit,
                                                                                      max_attempts=10,
                                                                                      max_iters=1000, restarts=0,
                                                                                      init_state=None, curve=True,
                                                                                      random_state=random_state)

    plt.cla()
    plt.plot(range(0, len(list(fitness_curve_HC))), -fitness_curve_HC[..., 0], '-')
    plt.title("default RHC fitness curve")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("./output3/RHC/1.jpg")

    # 调restart
    best_state_HC5, best_fitness_HC5, fitness_curve_HC5 = mlrose_hiive.random_hill_climb(problem_no_fit,
                                                                                         max_attempts=10,
                                                                                         max_iters=1000, restarts=5,
                                                                                         init_state=None, curve=True,
                                                                                         random_state=4)
    best_state_HC10, best_fitness_HC10, fitness_curve_HC10 = mlrose_hiive.random_hill_climb(
        problem_no_fit,
        max_attempts=10,
        max_iters=1000, restarts=10,
        init_state=None, curve=True,
        random_state=2)


    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(range(0, len(list(fitness_curve_HC))), -fitness_curve_HC[..., 0], '-',
            label='restarts = 0')  # Plot some data on the axes.
    ax.plot(range(0, len(list(fitness_curve_HC5))), -fitness_curve_HC5[..., 0], '-',
            label='restarts = 5')  # Plot more data on the axes...
    ax.plot(range(0, len(list(fitness_curve_HC10))), -fitness_curve_HC10[..., 0], '-',
            label='restarts = 10')  # ... and some more.
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("TSP: RHC behaviours with different restarts")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    fig.savefig("./output3/RHC/2.jpg")
    print(best_fitness_HC5)
    print(best_fitness_HC10)
    print(type(fitness_curve_HC10))
    return fitness_curve_HC5


def SA(problem_no_fit, random_state):
    print("using SA algorithm")
    # Solve using simulated annealing - attempt 1
    schedule_SA = mlrose_hiive.GeomDecay(init_temp=10, decay=0.95, min_temp=1)
    best_state_SA, best_fitness_SA, fitness_curve_SA = mlrose_hiive.simulated_annealing(problem_no_fit,
                                                                                        schedule=schedule_SA,
                                                                                        max_attempts=10, max_iters=1000,
                                                                                        init_state=None,
                                                                                        curve=True, random_state=42)

    # tune temperature
    schedule_arith = mlrose_hiive.ArithDecay(init_temp=10, decay=0.95, min_temp=1)
    best_state_SAarith, best_fitness_SAarith, fitness_curve_SAarith = mlrose_hiive.simulated_annealing(problem_no_fit,
                                                                                                       schedule=schedule_arith,
                                                                                                       max_attempts=10,
                                                                                                       max_iters=1000,
                                                                                                       init_state=None,
                                                                                                       curve=True,
                                                                                                       random_state=1)

    schedule_exp = mlrose_hiive.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
    best_state_SAexp, best_state_SAexp, fitness_curve_SAexp = mlrose_hiive.simulated_annealing(problem_no_fit,
                                                                                               schedule=schedule_exp,
                                                                                               max_attempts=10,
                                                                                               max_iters=1000,
                                                                                               init_state=None,
                                                                                               curve=True,
                                                                                               random_state=2)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(range(0, len(list(fitness_curve_SA))), -fitness_curve_SA[..., 0], '-',
            label='schedule = GeomDecay')  # Plot some data on the axes.
    ax.plot(range(0, len(list(fitness_curve_SAarith))), -fitness_curve_SAarith[..., 0], '-',
            label='schedule = ArithDecay')  # Plot more data on the axes...
    ax.plot(range(0, len(list(fitness_curve_SAexp))), -fitness_curve_SAexp[..., 0], '-',
            label='schedule = ExpDecay')  # ... and some more.
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("TSP: SA behaviour with different ways of temperature decay")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("./output3/SA/1.jpg")

    # tune temperature decay
    schedule_arith2 = mlrose_hiive.ArithDecay(init_temp=10, decay=0.5, min_temp=1)
    best_state_SAarith2, best_fitness_SAarith2, fitness_curve_SAarith2 = mlrose_hiive.simulated_annealing(
        problem_no_fit, schedule=schedule_arith2, max_attempts=10,
        max_iters=1000, init_state=None, curve=True,
        random_state=2)
    schedule_arith3 = mlrose_hiive.ArithDecay(init_temp=10, decay=0.1, min_temp=1)
    best_state_SAarith3, best_fitness_SAarith3, fitness_curve_SAarith3 = mlrose_hiive.simulated_annealing(
        problem_no_fit, schedule=schedule_arith3, max_attempts=10,
        max_iters=1000, init_state=None, curve=True,
        random_state=1)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(range(0, len(list(fitness_curve_SA))), -fitness_curve_SA[..., 0], '-',
            label='decay = 0.95')  # Plot some data on the axes.
    ax.plot(range(0, len(list(fitness_curve_SAarith2))), -fitness_curve_SAarith2[..., 0], '-',
            label='decay = 0.5')  # Plot more data on the axes...
    ax.plot(range(0, len(list(fitness_curve_SAarith3))), -fitness_curve_SAarith3[..., 0], '-',
            label='decay = 0.1')  # ... and some more.
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("TSP: SA behaviour with different temperature decay")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("./output3/SA/2.jpg")
    return fitness_curve_SAarith3


def GA(problem_no_fit, random_state):
    print("using GA algorithm")
    # Solve using genetic algorithm
    best_state_GA, best_fitness_GA, fitness_curve_GA = mlrose_hiive.genetic_alg(problem_no_fit, mutation_prob=0.2,
                                                                                max_attempts=100, curve=True,
                                                                                random_state=2)

    best_state_GA100, best_fitness_GA100, fitness_curve_GA100 = mlrose_hiive.genetic_alg(problem_no_fit,
                                                                                         mutation_prob=0.2,
                                                                                         pop_size=100,
                                                                                         max_attempts=100, curve=True,
                                                                                         random_state=2)
    best_state_GA300, best_fitness_GA300, fitness_curve_GA300 = mlrose_hiive.genetic_alg(problem_no_fit,
                                                                                         mutation_prob=0.2,
                                                                                         pop_size=300,
                                                                                         max_attempts=100, curve=True,
                                                                                         random_state=2)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(range(0, len(list(fitness_curve_GA))), -fitness_curve_GA[..., 0], '-',
            label='pop_size = 200')  # Plot some data on the axes.
    ax.plot(range(0, len(list(fitness_curve_GA100))), -fitness_curve_GA100[..., 0], '-',
            label='pop_size = 100')  # Plot more data on the axes...
    ax.plot(range(0, len(list(fitness_curve_GA300))), -fitness_curve_GA300[..., 0], '-',
            label='pop_size = 300')  # ... and some more.
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("TSP: GA behaviour with different population")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    fig.savefig("./output3/GA/1.jpg")
    return fitness_curve_GA


def MIMIC(problem_no_fit, random_state):
    print("using MIMIC algorithm")

    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose_hiive.mimic(problem_no_fit, pop_size=200,
                                                                                   keep_pct=0.2, max_attempts=10,
                                                                                   max_iters=1000, curve=True,
                                                                                   random_state=1)

    best_state_mimic100, best_fitness_mimic100, fitness_curve_mimic100 = mlrose_hiive.mimic(problem_no_fit,
                                                                                            pop_size=100, keep_pct=0.2,
                                                                                            max_attempts=10,
                                                                                            max_iters=1000, curve=True,
                                                                                            random_state=2)
    best_state_mimic300, best_fitness_mimic300, fitness_curve_mimic300 = mlrose_hiive.mimic(problem_no_fit,
                                                                                            pop_size=300, keep_pct=0.2,
                                                                                            max_attempts=10,
                                                                                            max_iters=1000, curve=True,
                                                                                            random_state=3)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(range(0, len(list(fitness_curve_mimic))), -fitness_curve_mimic[..., 0], '-',
            label='pop_size = 200')  # Plot some data on the axes.
    ax.plot(range(0, len(list(fitness_curve_mimic100))), -fitness_curve_mimic100[..., 0], '-',
            label='pop_size = 100')  # Plot more data on the axes...
    ax.plot(range(0, len(list(fitness_curve_mimic300))), -fitness_curve_mimic300[..., 0], '-',
            label='pop_size = 300')  # ... and some more.
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("MIMIC behaviour with different population")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    fig.savefig("./output3/MIMIC/1.jpg")
    return fitness_curve_mimic


def CompareModel(a, b, c, d):
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(range(0, len(list(a))), -a[..., 0], '-',
            label='HC')  # Plot some data on the axes.
    ax.plot(range(0, len(list(b))), -b[..., 0], '-',
            label='SA')  # Plot more data on the axes...
    ax.plot(range(0, len(list(c))), -c[..., 0], '-', label='GA')  # ... and some more.
    ax.plot(range(0, len(list(d))), -d[..., 0], '-', label='MIMIC')
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("TSP fitness curves using 4 RO algorithms")  # Add a title to the axes.
    ax.legend()
    fig.savefig("./output3/C/2.jpg")

def CompareTime():
    startHC10 = time.time()
    best_state_HC10, best_fitness_HC10, fitness_curve_HC10 = mlrose_hiive.random_hill_climb(problem_no_fit,
                                                                                            max_attempts=10,
                                                                                            max_iters=1000, restarts=10,
                                                                                            init_state=None, curve=True,
                                                                                            random_state=42)
    endHC10 = time.time()
    HC10time = endHC10 - startHC10
    print(HC10time)

    startSAarith3 = time.time()
    schedule_arith3 = mlrose_hiive.ArithDecay(init_temp=10, decay=0.1, min_temp=1)
    best_state_SAarith3, best_fitness_SAarith3, fitness_curve_SAarith3 = mlrose_hiive.simulated_annealing(
        problem_no_fit, schedule=schedule_arith3, max_attempts=10,
        max_iters=1000, init_state=None, curve=True,
        random_state=1)
    endSAarith3 = time.time()
    SAarith3 = endSAarith3 - startSAarith3
    print(SAarith3)

    startGA = time.time()
    best_state_GA, best_fitness_GA, fitness_curve_GA = mlrose_hiive.genetic_alg(problem_no_fit, mutation_prob=0.2,
                                                                                max_attempts=100, curve=True,
                                                                                random_state=2)
    endGA = time.time()
    GA = endGA - startGA
    print(GA)

    startMIMIC = time.time()
    best_state_mimic300, best_fitness_mimic300, fitness_curve_mimic300 = mlrose_hiive.mimic(problem_no_fit,
                                                                                            pop_size=300, keep_pct=0.2,
                                                                                            max_attempts=10,
                                                                                            max_iters=1000, curve=True,
                                                                                            random_state=3)
    endMIMIC = time.time()
    MIMIC = endMIMIC - startMIMIC
    print(MIMIC)

    runtime_list = [HC10time, SAarith3, GA, MIMIC]
    print(runtime_list)
    names = ["RHC", "SA", "GA", "MIMIC"]
    print(names)
    fig = plt.figure(figsize=(8, 5))

    # creating the bar plot
    plt.bar(names, runtime_list, width=0.4)
    plt.ylabel("Seconds")
    plt.title("Running Time")
    plt.savefig("./output3/C/1.jpg")


if __name__ == '__main__':
    print("this is TSP:")
    # Define optimization problem object
    number_of_cities = 20
    problem_no_fit = TSPGenerator().generate(seed=322, number_of_cities=number_of_cities)
    BEST_RHC = RHC(problem_no_fit, 42)
    BEST_SA = SA(problem_no_fit, 42)
    BEST_GA = GA(problem_no_fit, 42)
    BEST_MIMIC = MIMIC(problem_no_fit, 42)
    CompareTime()
    CompareModel(BEST_RHC, BEST_SA, BEST_GA, BEST_MIMIC)