import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import os
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
from sklearn.metrics import classification_report

def run_pso_optimization(probs_list, true_labels, num_particles=30, max_iter=100):
    """
    Otimiza pesos usando Particle Swarm Optimization para combinar vetores de probabilidade.

    Returns:
        np.ndarray: Pesos normalizados otimizados
    """
    num_clients = len(probs_list)
    num_samples = len(true_labels)

    def loss(weights):
        weights = np.clip(weights, 0, None)
        weights = weights / np.sum(weights)
        combined = np.zeros_like(probs_list[0])
        for i in range(num_clients):
            combined += weights[i] * np.array(probs_list[i])
        predicted = np.argmax(combined.reshape((num_samples, -1)), axis=1)
        return -accuracy_score(true_labels, predicted)  # minimizar negativo da acurácia

    bounds = [(0, 1)] * num_clients

    result = differential_evolution(loss, bounds, strategy='best1bin', maxiter=max_iter, popsize=num_particles, tol=1e-6, seed=42)
    
    best_weights = result.x / np.sum(result.x)
    print(f"[PSO] Melhor acurácia encontrada: {-result.fun:.4f}")
    return best_weights

def run_hybrid_ensemble_ga_stacking(probs_list, true_labels, ga_weights):
    """
    Executa ensemble híbrido: primeiro combina os vetores via GA, depois aplica Stacking com Logistic Regression.
    
    Args:
        probs_list (list of np.ndarray): Vetores de probabilidade dos modelos.
        true_labels (np.ndarray): Rótulos verdadeiros.
        ga_weights (np.ndarray): Pesos aprendidos pelo GA.
        
    Returns:
        float: Acurácia final com ensemble híbrido.
    """
    # Etapa 1: combinação ponderada via GA
    combined_probs_ga = np.zeros_like(probs_list[0])
    for i, probs in enumerate(probs_list):
        combined_probs_ga += ga_weights[i] * np.array(probs)

    # Etapa 2: usar o vetor combinado como entrada do stacking
    # Cada amostra agora é um vetor de tamanho num_classes
    X_stacking = combined_probs_ga
    y_stacking = true_labels

    # Treinar meta-classificador
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_stacking, y_stacking)

    # Predição final
    final_preds = meta_model.predict(X_stacking)
    final_acc = accuracy_score(true_labels, final_preds)

    print(f"[HÍBRIDO GA + STACKING] Acurácia final: {final_acc:.4f}")
    return final_acc, X_stacking, y_stacking

def run_genetic_algorithm(probs_list, true_labels, population_size=40, generations=100, mutation_rate=0.3):
    """
    Otimiza pesos para combinação de vetores de probabilidade usando algoritmo genético com melhorias.
    """
    num_clients = len(probs_list)
    num_samples = len(true_labels)

    print("Running GA...")
    print(f"[GA] Rótulos únicos: {set(true_labels)}")

    # Acurácia de ensemble simples
    avg_probs = np.mean(probs_list, axis=0)
    avg_preds = np.argmax(avg_probs, axis=1)
    acc_base = accuracy_score(true_labels, avg_preds)
    print(f"[BASELINE] Média simples das probabilidades => Accuracy: {acc_base:.4f}")

    # Geração inicial aleatória com mais diversidade
    def random_weights():
        return np.random.dirichlet(np.ones(num_clients))

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
                child[mutation_idx] += np.random.normal(0, 0.3)
                child = np.clip(child, 0, None)

            child = child / np.sum(child)
            new_population.append(child)

        # Injetar novos indivíduos aleatórios para diversidade
        if gen % 10 == 0:
            for _ in range(3):
                new_population[-1 - _] = random_weights()

        population = new_population

    best_individual = max(population, key=fitness)

    # Diretório de saída
    out_dir = f"results/ga"
    os.makedirs(out_dir, exist_ok=True)

    # Plot: evolução da acurácia
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, generations + 1), best_scores, marker='o')
    plt.title("Evolução da Acurácia - Algoritmo Genético")
    plt.xlabel("Geração")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ga_accuracy_evolution.png")
    plt.close()

    # Plot: pesos aprendidos
    plt.figure(figsize=(8, 5))
    plt.bar(range(num_clients), best_individual)
    plt.title("Pesos aprendidos pelo GA")
    plt.xlabel("Cliente")
    plt.ylabel("Peso")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ga_weights_final.png")
    plt.close()

    return best_individual

def run_hybrid_ensemble_pso_stacking(probs_list, true_labels, pso_weights):
    """
    Executa ensemble híbrido: primeiro combina os vetores via PSO, depois aplica Stacking com Logistic Regression.
    
    Args:
        probs_list (list of np.ndarray): Vetores de probabilidade dos modelos.
        true_labels (np.ndarray): Rótulos verdadeiros.
        pso_weights (np.ndarray): Pesos aprendidos pelo PSO.
        
    Returns:
        float: Acurácia final com ensemble híbrido.
        np.ndarray: Matriz combinada (X).
        np.ndarray: Vetor de rótulos (y).
    """
    # Etapa 1: combinação ponderada via PSO
    combined_probs_pso = np.zeros_like(probs_list[0])
    for i, probs in enumerate(probs_list):
        combined_probs_pso += pso_weights[i] * np.array(probs)

    # Etapa 2: usar o vetor combinado como entrada do stacking
    X_stacking = combined_probs_pso
    y_stacking = true_labels

    # Treinar meta-classificador
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_stacking, y_stacking)

    # Predição final
    final_preds = meta_model.predict(X_stacking)
    final_acc = accuracy_score(true_labels, final_preds)

    print(f"[HÍBRIDO PSO + STACKING] Acurácia final: {final_acc:.4f}")
    return final_acc, X_stacking, y_stacking

def evaluate_weighted_probs(probs_list, weights, true_labels, method):
    """
    Combina vetores de probabilidade usando pesos e calcula a acurácia final.
    """
    combined = np.zeros_like(probs_list[0])
    for i, probs in enumerate(probs_list):
        combined += weights[i] * np.array(probs)

    preds = np.argmax(combined, axis=1)
    acc = accuracy_score(true_labels, preds)

    #Saving classification report
    report = classification_report(true_labels, preds, digits=4)
    report_path = f"results/{method}/classification_report_round.txt"
    with open(report_path, "w") as f:
        f.write(f"Classification Report\n\n")
        f.write(report) 
    

    return acc
