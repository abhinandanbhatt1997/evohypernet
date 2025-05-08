import torch
from evolution.engine import EvolutionEngine

def run_evolution(generations=5):
    engine = EvolutionEngine(population_size=6, mutation_rate=0.4,
                             device="cuda" if torch.cuda.is_available() else "cpu")

    for gen in range(generations):
        print(f"\n🌱 Generation {gen}")
        engine.evolve()
        for i, (dna, score) in enumerate(zip(engine.population, engine.fitness_scores)):
            print(f"#{i}: {dna} | Fitness (accuracy): {score:.4f}")

if __name__ == "__main__":
    run_evolution()
