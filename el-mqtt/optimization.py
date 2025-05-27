import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import os
import argparse

def run_genetic_algorithm(probs_list, true_labels, population_size=20, generations=50, mutation_rate=0.1):
    """
    Otimiza pesos para combinação de vetores de probabilidade usando algoritmo genético com perturbação e log.
    """
    num_clients = len(probs_list)
    num_samples = len(true_labels)

    print("Running GA...")

    # Geração inicial aleatória (pesos somando 1)
    def random_weights():
        w = np.random.rand(num_clients)
        return w / np.sum(w)

    population = [random_weights() for _ in range(population_size)]

    def fitness(weights):
        combined = np.zeros_like(probs_list[0])
        for i in range(num_clients):
            combined += weights[i] * np.array(probs_list[i])
        predicted = np.argmax(combined.reshape((num_samples, -1)), axis=1)
        return accuracy_score(true_labels, predicted)

    best_scores = []

    for gen in range(generations):
        scored_population = [(ind, fitness(ind)) for ind in population]
        scored_population.sort(key=lambda x: x[1], reverse=True)

        best_score = scored_population[0][1]
        best_scores.append(best_score)
        print(f"[Gen {gen+1}] Best Accuracy: {best_score:.4f}")

        elites = [ind for ind, _ in scored_population[:5]]
        new_population = elites[:]

        while len(new_population) < population_size:
            p1, p2 = random.sample(elites, 2)
            crossover_point = random.randint(1, num_clients - 1)
            child = np.concatenate((p1[:crossover_point], p2[crossover_point:]))

            if random.random() < mutation_rate:
                mutation_idx = random.randint(0, num_clients - 1)
                child[mutation_idx] += np.random.normal(0, 0.1)
                child = np.clip(child, 0, None)

            child = child / np.sum(child)
            new_population.append(child)

        population = new_population

    best_individual = max(population, key=fitness)

    os.makedirs("results", exist_ok=True)
    if args.ensemble_method == "ga":   # Se for GA, criar pasta específica
        os.makedirs("results/ga", exist_ok=True)
    elif args.ensemble_method == "stacking":  # Se for Stacking, criar pasta específica
        os.makedirs("results/stacking", exist_ok=True)
    elif args.ensemble_method == "voting":  # Se for Voting, criar pasta específica
        os.makedirs("results/voting", exist_ok=True)
    else:
        os.makedirs("results/baseline", exist_ok=True)

    # Plot evolução da acurácia
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, generations + 1), best_scores, marker='o')
    plt.title("Evolução da Acurácia - Algoritmo Genético")
    plt.xlabel("Geração")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/{args.ensemble_method}/ga_accuracy_evolution.png")

    return best_individual


def evaluate_weighted_probs(probs_list, weights, true_labels):
    """
    Combina vetores de probabilidade usando pesos e calcula a acurácia final.
    """
    combined = np.zeros_like(probs_list[0])
    for i, probs in enumerate(probs_list):
        combined += weights[i] * np.array(probs)

    preds = np.argmax(combined, axis=1)
    acc = accuracy_score(true_labels, preds)
    return acc