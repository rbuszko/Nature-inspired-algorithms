import random
import numpy as np
import matplotlib.pyplot as plt
from random import sample, randint
from tabulate import tabulate
import time
import bee
import genetic

def generate_data(amount, max_weight, max_value):
    random_weights = [randint(1, max_weight) for i in range(amount)]  # sample(range(1, max_weight), amount)
    random_values = [randint(1, max_value) for i in range(amount)]

    ids = np.linspace(0, amount - 1, amount)
    weights = np.array(random_weights)
    values = np.array(random_values)

    return np.concatenate(
        (ids[:, np.newaxis], weights[:, np.newaxis], values[:, np.newaxis]), axis=1)


def knapsack_greedy(C, data):
    score = data[:, 2] / data[:, 1]
    data = np.concatenate((data, score[:, np.newaxis]), axis=1)
    data = data[data[:, 3].argsort()][::-1]
    lim = np.shape(data)[0]
    i = 0
    backpack = []
    while i < lim:
        if data[i, 1] <= C:
            C = C - data[i, 1]
            backpack.append(data[i, :])
            i = i + 1
        else:
            i = i + 1
    np_b = np.array(backpack)
    if np_b.any():
        return np.sum(np_b[:, 2]), len(backpack)
    return 0, 0


def knapsack_brute(C, w, v, n):
    # initial conditions
    if n == 0 or C == 0:
        return 0
    # If weight is higher than capacity then it is not included
    if w[n - 1] > C:
        return knapsack_brute(C, w, v, n - 1)
    # return either nth item being included ozr not
    else:
        return max(v[n - 1] + knapsack_brute(C - w[n - 1], w, v, n - 1), knapsack_brute(C, w, v, n - 1))


##########################################################
# Tworzenie / wczytywanie danych / określenie parametrów #
##########################################################
datasets_number = 4
base = 10
KNAPSACK_CAPACITY_MODIFIER = [0.2, 0.35, 0.5]
max_weight = 50
max_value = 50
datasets = []
knapsack_capacities = []
problem_names = []

random.seed(241525)

for number in range(1, datasets_number + 1):
    problem_names.append(f"Rand {pow(base, number+1)}")
    datasets.append(generate_data(pow(base, number+1), max_weight, max_value))
    tmp_cap = []
    for x in KNAPSACK_CAPACITY_MODIFIER:
        tmp_cap.append(np.round(np.sum(datasets[number-1][:, 1]) * x))
    knapsack_capacities.append(tmp_cap)

#####################################
# Pomiar czasu działania algorytmów #
#####################################
# Zwykły
times = []
summary_values = []
knapsacks_item_number = []
# Parametry
swarm_size = 100
number_of_cycles = 300
limit = int(swarm_size/10)

# Referencyjny
times_ref = []
summary_values_ref = []
knapsacks_item_number_ref = []

for number in range(datasets_number):
    # Zwykły
    tmp_value = []
    tmp_time = []
    tmp_number = []

    # Referencyjny
    tmp_value_ref = []
    tmp_time_ref = []
    tmp_number_ref = []

    for cap in knapsack_capacities[number]:
        # Zwykly
        start = time.time()
        ABC = bee.Hive(swarm_size=swarm_size, data=datasets[number], capacity=cap)
        opt_val, number_of_items = ABC.run(number_of_cycles=number_of_cycles, limit=limit)
        end = time.time()

        tmp_value.append(opt_val)
        tmp_time.append(end - start)
        tmp_number.append(number_of_items)

        # Referencyjny
        print("check")
        start = time.time()
        opt_val_ref, number_of_items_ref = knapsack_greedy(cap, datasets[number])
        end = time.time()

        tmp_value_ref.append(opt_val_ref)
        tmp_number_ref.append(number_of_items_ref)
        tmp_time_ref.append(end - start)

    # Zwykły
    summary_values.append(tmp_value)
    times.append(tmp_time)
    knapsacks_item_number.append(tmp_number)

    # Referencyjny
    summary_values_ref.append(tmp_value_ref)
    times_ref.append(tmp_time_ref)
    knapsacks_item_number_ref.append(tmp_number_ref)

plt.plot([len(data) for data in datasets], [time[0] for time in times], 'k--', linewidth=1.0)
plt.plot([len(data) for data in datasets], [time[1] for time in times], 'b--', linewidth=1.0)
plt.plot([len(data) for data in datasets], [time[2] for time in times], 'r--', linewidth=1.0)

plt.plot([len(data) for data in datasets], [time[0] for time in times_ref], 'k', linewidth=1.0)
plt.plot([len(data) for data in datasets], [time[1] for time in times_ref], 'b', linewidth=1.0)
plt.plot([len(data) for data in datasets], [time[2] for time in times_ref], 'r', linewidth=1.0)


plt.title('Wykres obrazujący złożoność czasową algorytmów')
plt.xlabel('Liczebność obiektów w zbiorze danych wejśćiowych')
plt.ylabel('Czas [s]')
plt.legend(['20% wagi', '35% wagi', '50% wagi'])
plt.savefig('fig.png')


result = []
result20 = []
result35 = []
result50 = []
for i in range(len(problem_names)):
    result.append([problem_names[i]
                   , knapsacks_item_number[i][0], summary_values[i][0]
                   , knapsacks_item_number[i][1], summary_values[i][1]
                   , knapsacks_item_number[i][2], summary_values[i][2]])
    result20.append([problem_names[i]
                   , knapsacks_item_number[i][0], summary_values[i][0]
                   , knapsacks_item_number_ref[i][0], summary_values_ref[i][0]])
    result35.append([problem_names[i]
                   , knapsacks_item_number[i][1], summary_values[i][1]
                   , knapsacks_item_number_ref[i][1], summary_values_ref[i][1]])
    result50.append([problem_names[i]
                   , knapsacks_item_number[i][2], summary_values[i][2]
                   , knapsacks_item_number_ref[i][2], summary_values_ref[i][2]])

print("* Knapsack bee algorithm result for diff capacities *")
print(tabulate([["Problem name", "No. Items 20%", "Sum Profits 20%",
                 "No. Items 35%", "Sum Profits 35%", "No. Items 50%", "Sum Profits 50%"]]
               + result, headers='firstrow', tablefmt='fancy_grid'))

print("* Algorithm comparison for capacity 20% *")
print(tabulate([["Problem name", "No. Items", "Sum Profits",
                 "No. Items", "Sum Profits"]] + result20, headers='firstrow', tablefmt='fancy_grid'))

print("* Algorithm comparison for capacity 35% *")
print(tabulate([["Problem name", "No. Items", "Sum Profits",
                 "No. Items", "Sum Profits"]] + result35, headers='firstrow', tablefmt='fancy_grid'))

print("* Algorithm comparison for capacity 50% *")
print(tabulate([["Problem name", "No. Items", "Sum Profits",
                 "No. Items", "Sum Profits"]] + result50, headers='firstrow', tablefmt='fancy_grid'))

print(np.sum(times))
print(np.sum(times_ref))
