import numpy as np
import random
import copy


class Bee:
    def __init__(self):
        self.solution = None
        self.function_value = 0
        self.fitness = 0
        self.trial = 0
        self.probability = 0
        self.weight = 0

    def discard_items_over_capacity(self, data, capacity, item_ranking):
        weights = data[:, 1]
        current_weight = np.sum(self.solution * weights)
        for index, probability in item_ranking:
            if current_weight > capacity:
                if self.solution[int(index)] == 1:
                    self.solution[int(index)] = 0
                    current_weight -= weights[int(index)]
            else:
                self.weight = current_weight
                break

    def set_random_solution(self, data, capacity, item_ranking):
        self.solution = np.array([random.randint(0, 1) for i in range(len(data))])
        self.discard_items_over_capacity(data, capacity, item_ranking)

    def calculate_function_value(self, data):
        self.function_value = np.sum(self.solution * data[:, 2])

    def calculate_fitness(self):
        if self.function_value >= 0:
            self.fitness = 1 / (1 + self.function_value)
        else:
            self.fitness = 1 + abs(self.function_value)

    def calcutale_probability(self, all_probabilities):
        self.probability = self.function_value / all_probabilities

    def add_to_max(self, data, item_ranking, capacity):
        weights = data[:, 1]
        current_weight = np.sum(self.solution * weights)
        for index, probability in item_ranking:
            if self.solution[int(index)] == 0:
                if current_weight + weights[int(index)] <= capacity:
                    self.solution[int(index)] = 1
                    current_weight += weights[int(index)]

        self.calculate_function_value(data)
        self.calculate_fitness()


class Hive:
    def __init__(self, swarm_size, data, capacity):
        score = data[:, 2] / data[:, 1]
        probability_of_items = capacity / np.sum(data[:, 1]) * score / np.mean(score)
        probability_of_items /= (np.sum(probability_of_items))
        probability_of_items = np.concatenate((data[:, 0, np.newaxis], probability_of_items[:, np.newaxis]), axis=1)
        item_ranking = probability_of_items[probability_of_items[:, 1].argsort()]

        self._swarm_size = swarm_size
        self._item_ranking = item_ranking
        self._data = data
        self._capacity = capacity
        self.best_food_source = None
        self.best_function_value = 0
        self._hive = [Bee() for i in range(round(swarm_size / 2))]
        self.weight = 0

        # self.greed = self.init_greed_solution(self._data,self._capacity)

        for bee in self._hive:
            bee.set_random_solution(self._data, self._capacity, self._item_ranking)
            bee.calculate_function_value(self._data)
            bee.calculate_fitness()

    def employed_bee_phase(self):
        for bee in self._hive:
            new_bee = copy.deepcopy(bee)
            index = random.choice(np.argwhere(new_bee.solution == 0).reshape(-1))
            new_bee.solution[index] = 1
            new_bee.discard_items_over_capacity(data=self._data, capacity=self._capacity,
                                                item_ranking=self._item_ranking)
            new_bee.calculate_function_value(data=self._data)
            new_bee.calculate_fitness()
            if bee.fitness > new_bee.fitness:
                bee = new_bee
                bee.trial = 0
            else:
                bee.trial += 1
        # print("debug")

    def onlooker_bee_phase(self):
        function_value_sum = sum([bee.function_value for bee in self._hive])
        for bee in self._hive:
            bee.calcutale_probability(function_value_sum)

        number_of_onlooker_bees = int(self._swarm_size / 2)
        while number_of_onlooker_bees > 0:
            for bee in self._hive:
                if number_of_onlooker_bees < 1:
                    break
                r = random.uniform(0, 1)
                if r < bee.probability:
                    new_bee = copy.deepcopy(bee)
                    index = random.choice(np.argwhere(new_bee.solution == 0).reshape(-1))
                    new_bee.solution[index] = 1
                    new_bee.discard_items_over_capacity(data=self._data, capacity=self._capacity,
                                                        item_ranking=self._item_ranking)
                    new_bee.calculate_function_value(data=self._data)
                    new_bee.calculate_fitness()
                    if bee.fitness > new_bee.fitness:
                        bee = new_bee
                        bee.trial = 0
                    else:
                        bee.trial += 1
                    number_of_onlooker_bees -= 1

        for bee in self._hive:
            if bee.function_value > self.best_function_value:
                # print("XXXXXXXXXXXXXXXXXXX")
                # print("XXXXXXXXXXXXXXXXXXX")
                # print(bee.solution)
                # bee.add_to_max(self._data, self._item_ranking, self._capacity)
                # print(bee.solution)
                self.best_function_value = bee.function_value
                self.best_food_source = bee.solution
                self.weight = bee.weight

    def scout_bee_phase(self, limit):
        for bee in self._hive:
            if bee.trial > limit:
                bee.set_random_solution(self._data, self._capacity, self._item_ranking)
                bee.calculate_function_value(self._data)
                bee.calculate_fitness()

    def run(self, number_of_cycles, limit):
        print("MAX CAPACITY")
        print(self._capacity)
        for i in range(number_of_cycles):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase(limit)
            # print("BEST SO FAR:",self.best_function_value)
        print(np.sum(self.best_food_source * self._data[:, 1]))
        print("Hello from bee")
        # print(data.T)
        # print("Best food source")
        # print(f" {self.best_food_source}")
        # print(f"with highest value in backpack: {self.best_function_value}")
        # print(f"with weight: {self.weight}")
        return self.best_function_value, np.count_nonzero(self.best_food_source)

    def init_greed_solution(self, data, C):
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
        food_source = np.linspace(0,0,lim)
        for i in range(lim):
            food_source[int(np_b[i,0])]=1
        return food_source

def generate_data(amount, max_weight, max_value):
    random_weights = [random.randint(1, max_weight) for i in range(amount)]  # sample(range(1, max_weight), amount)
    random_values = [random.randint(1, max_value) for i in range(amount)]

    ids = np.linspace(0, amount - 1, amount).astype(int)
    weights = np.array(random_weights)
    values = np.array(random_values)

    return np.concatenate(
        (ids[:, np.newaxis], weights[:, np.newaxis], values[:, np.newaxis]), axis=1)

if __name__ == "__main__":
    number_of_items = 100
    data = generate_data(number_of_items, 10, 10)
    capacity = round(0.35 * np.round(np.sum(data[:, 1])))
    swarm_size = 200
    print(capacity)
    # print(data.T)

    ABC = Hive(swarm_size=swarm_size, data=data, capacity=capacity)
    print(ABC.run(number_of_cycles=100, limit=10))