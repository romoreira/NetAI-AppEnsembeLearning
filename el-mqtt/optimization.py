import numpy as np
from sklearn.metrics import accuracy_score
import random


def run_genetic_algorithm(probs_list, true_labels, population_size=20, generations=50, mutation_rate=0.1):
    """
    Otimiza pesos para combinação de vetores de probabilidade usando algoritmo genético.

    Args:
        probs_list (list of np.ndarray): lista de vetores de probabilidade de diferentes clientes.
        true_labels (np.ndarray): rótulos verdadeiros.
        population_size (int): número de indivíduos na população.
        generations (int): número de gerações.
        mutation_rate (float): taxa de mutação.

    Returns:
        np.ndarray: vetor de pesos normalizado que maximizou a acurácia.
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
        # Combina os vetores de probabilidade usando pesos
        combined = np.zeros_like(probs_list[0])
        for i in range(num_clients):
            combined += weights[i] * np.array(probs_list[i])
        predicted = np.argmax(combined.reshape((num_samples, -1)), axis=1)
        return accuracy_score(true_labels, predicted)

    for _ in range(generations):
        # Avaliar fitness
        scored_population = [(ind, fitness(ind)) for ind in population]
        scored_population.sort(key=lambda x: x[1], reverse=True)

        # Seleção dos melhores (elitismo + roleta)
        elites = [ind for ind, score in scored_population[:5]]
        new_population = elites[:]

        while len(new_population) < population_size:
            # Seleção por torneio
            p1, p2 = random.sample(elites, 2)
            crossover_point = random.randint(1, num_clients - 1)
            child = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
            if random.random() < mutation_rate:
                mutation_idx = random.randint(0, num_clients - 1)
                child[mutation_idx] = random.random()
            child = child / np.sum(child)  # normalizar
            new_population.append(child)

        population = new_population

    # Melhor indivíduo após todas as gerações
    best_individual = max(population, key=fitness)
    return best_individual

def evaluate_weighted_probs(probs_list, weights, true_labels):
    """
    Combina vetores de probabilidade usando pesos e calcula a acurácia final.

    Args:
        probs_list (list of np.ndarray): Lista com vetores de probabilidade (n_amostras x n_classes).
        weights (np.ndarray): Pesos normalizados (ex: [0.4, 0.6]).
        true_labels (np.ndarray): Rótulos verdadeiros (ex: [0, 1, 2, ...]).

    Returns:
        float: Acurácia final após combinação.
    """
    combined = np.zeros_like(probs_list[0])
    for i, probs in enumerate(probs_list):
        combined += weights[i] * np.array(probs)
    
    preds = np.argmax(combined, axis=1)
    acc = accuracy_score(true_labels, preds)
    return acc