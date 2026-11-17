# Baseline GA: fitness is the exact count of hydrophobic residues (no ESM-2, no head).
# Example run (no seed): the population mean goes from ~13/27 hydrophobic residues at generation 0 to ~26.9/27 around generation 17, then stays there.

import random
from deap import base, creator, tools

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

SEQ_LENGTH = 27
CXPB = 0.5
MTPB = 0.2
NGEN = 50

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_amino_acid", random.choice, amino_acids)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_amino_acid, n=SEQ_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOnePoint)

def mutate_amino_acid(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(amino_acids)
    return individual,

toolbox.register("mutate", mutate_amino_acid, indpb=0.02)

def count_hydrophobic_residues(individual):
    hydrophobic_residues = ['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
    return sum(1 for aa in individual if aa in hydrophobic_residues),

toolbox.register("evaluate", count_hydrophobic_residues)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100)



for ind in population:
    ind.fitness.values = toolbox.evaluate(ind)
    

print(sum(sum(ind.fitness.values) for ind in population) / len(population))

for gen in range(NGEN):
    chosen = toolbox.select(population, len(population))
    offspring = [toolbox.clone(ind) for ind in chosen]



    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values


    for mutant in offspring:
        if random.random() < MTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)

    population[:] = offspring
    print(gen,sum(sum(ind.fitness.values) for ind in population) / len(population))

