
import time
import os
from math import sqrt
import argparse
import itertools
import random
import numpy as np

def euclidean_distance(point1, point2):
    """"
    Calculate distance among two points
    INPUT:
    Two points
    RETURN:
        Distance between two points
    """
    L1, x1, y1 = point1
    L2, x2, y2 = point2
    return round(sqrt((x2 - x1)**2 + (y2 - y1)**2))

def total_distance(tour):
    """
    :param tour: total distance for the pass point
    :return: total distance
    """
    dist = 0
    for i in range(len(tour) - 1):
        dist += euclidean_distance(tour[i], tour[i + 1])
    if tour:
        dist += euclidean_distance(tour[-1], tour[0])
    return dist

def brute_force_tsp(coordinates, cutoff_time):
    """
    :param coordinates: Location information included ID and 2-D
    :param cutoff_time: time set up
    :return:
        Best tour
    """
    start_time = time.time()
    best_tour = None
    best_distance = float('inf')

    for tour in itertools.permutations(coordinates):
        current_distance = total_distance(tour)
        if time.time() - start_time > cutoff_time:
            break
        if current_distance < best_distance:
            best_distance = current_distance
            best_tour = tour
    total_time = round((time.time() - start_time), 5)
    return best_tour, best_distance, total_time


def read_tsp_mst(file_path):
     """
     This function reads the tsp files
     Parameters
     ----------
     file_path : TSP file
          File name.
     Returns
     -------
     city_pos : list
          A list with all coordinates.
     """
     
     with open(file_path, 'r') as file:
         lines = file.readlines()
     
     # Find the section containing city coordinates
     node_coord_section_line = next(line for line in lines if line.startswith('NODE_COORD_SECTION'))
     node_coord_section_index = lines.index(node_coord_section_line) + 1
     
     # Extract city coordinates
     city_data = [line.split() for line in lines[node_coord_section_index:] if line.strip() and line != "EOF"]
     
     # Store coordinates in a list
     city_data.pop()
     city_pos = [(float(x), float(y)) for _, x, y in city_data]
     
     return city_pos



def find_mst(city_pos):
     """
     This function finds the MST given the city coordinates
     Parameters
     ----------
     city_pos : list
          A list of coordinates of the regions
     Returns
     -------
     edge_li : list
          A list of edges in the MST.
     """
     
     
     num_cities = len(city_pos)
     
     # A matrix storing all the distances
     dis_mat = []
     
     for i in range(num_cities):
          dis_li = []
          for j in range(num_cities):
               distance = (city_pos[i][0] - city_pos[j][0])**2 + \
                    (city_pos[i][1] - city_pos[j][1])**2
               
               distance = distance ** 0.5
               
               dis_li.append(distance)
               
          dis_mat.append(dis_li)
     
     edge_li = []
     visited_cities = [1]
     unvisited_cities = [_ for _ in range(2, num_cities+1)]
     
     while len(unvisited_cities) > 0:
          min_cost = float('inf')
          for i in visited_cities:
               for j in unvisited_cities:
                    if dis_mat[i-1][j-1] < min_cost:
                         min_cost = dis_mat[i-1][j-1] 
                         min_cost_edge = (i, j)
          edge_li.append(min_cost_edge)
          visited_cities.append(min_cost_edge[1])
          unvisited_cities.remove(min_cost_edge[1])
          
     return edge_li



def dfs(edge_li):
     """
     This function traverses a MST by DFS
     Parameters
     ----------
     edge_li : list
          A list store all the edges in the MST.
     Returns
     -------
     traverse_li : list
          The order of traverse
     """
     
     S = [1]  # the initialized stack
     traverse_li = []
     
     explored = {i: False for i in range(1, len(edge_li)+2)}
     
     while len(S) > 0:
          u = S.pop()
          if explored[u] == False:
               traverse_li.append(u)
               explored[u] = True
               
               for (i, j) in edge_li:
                    if i == u:
                         S.append(j)
                    elif j == u:
                         S.append(i)   
     
                         
     return traverse_li


def compute_final_cost_mst(traverse_li, city_pos):
     final_cost = 0
     city_num = len(traverse_li)
     for i in range(city_num):
          if i != city_num-1:
               distance = (city_pos[traverse_li[i]-1][0] - city_pos[traverse_li[i+1]-1][0])**2 + \
                    (city_pos[traverse_li[i]-1][1] - city_pos[traverse_li[i+1]-1][1])**2
               
               distance = distance ** 0.5
               final_cost += distance
          else:
               distance = (city_pos[traverse_li[i]-1][0] - city_pos[traverse_li[0]-1][0])**2 + \
                    (city_pos[traverse_li[i]-1][1] - city_pos[traverse_li[0]-1][1])**2
               
               distance = distance ** 0.5
               final_cost += distance
               
     return int(final_cost)

# 写你们的function：
    #def()

class LS():
    def __init__(self, seed, fname):
        self.seed = seed

        #parameters of simulated annealing
        self.T = 250 # initial temperature
        self.k = 100 # constant
        self.coolingFraction = 0.999 # fraction by which temperature is lowered
        self.M = 200 # temperature will be lowered every M steps
        self.nSteps = 2000000 # total number of steps
        self.stopping_temp = 1
        self.t = 0

        self.n = 0
        self.pos = None
        path = './DATA/'+fname
        with open(path) as f:
            LINES = f.readlines()
            for i, line in enumerate(LINES):
                if i == 2:
                    self.n = int(line.rstrip('\n').split(' ')[1])
                    self.pos = np.zeros((self.n,2))
                elif i >= 5 and i < 5+self.n:
                    temp = np.array([float(line.rstrip('\n').split(' ')[1]), float(line.rstrip('\n').split(' ')[2])]).reshape(1,-1)
                    self.pos[i-5,:] = temp
            f.close()    

        self.solution = list(range(self.n))
        random.seed(self.seed)
        self.solution = random.sample(self.solution, self.n)
        self.solution_new = self.solution.copy()

        self.cost = self.cost_func(self.solution)
        self.cost_new = self.cost*2
    
    def cost_func(self,sol):
        Sum = 0
        for i in range(1,len(sol)):
            Sum  += int(np.linalg.norm(self.pos[sol[i],:] - self.pos[sol[i-1],:]))
        Sum += int(np.linalg.norm(self.pos[sol[-1],:] - self.pos[sol[0],:]))
        return Sum
    
    def acceptance_probability(self):
        return np.exp(-abs(self.cost_new - self.cost) / (self.k*self.T))
    
    
    def swap_node_pair_at_random(self, sol):
        sol_new = sol.copy()
        random.seed(self.seed+self.t)
        i = random.sample(sol, 2)
        sol_new[i[0]], sol_new[i[1]] = sol[i[1]], sol[i[0]]

        return sol_new
    
    def cost_optimization(self):
        while self.t < self.nSteps and self.T > self.stopping_temp:
            self.solution_new = self.swap_node_pair_at_random(self.solution)
            self.cost_new = self.cost_func(self.solution_new)

            #compare the cost
            #if the cost is lower, just accept it
            if self.cost_new<=self.cost:
                self.solution = self.solution_new
                self.cost = self.cost_new
            else:
                #do the probability trick
                acceptance_rate = self.acceptance_probability()
                random.seed(self.seed+self.t)
                r = random.random()
                if r < acceptance_rate:
                    self.solution = self.solution_new
                    self.cost = self.cost_new

            if self.t % self.M == self.M-1:
                self.T *= self.coolingFraction
            self.t += 1
        return self.cost, self.solution


def main(tsp,algo,cutoff,seed):
    tsp_name = tsp.split('/')[-1].split('.')[0]

    sol_file = "_".join([tsp_name, algo, str(cutoff)]) + '.sol'
    output_dir = './output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if algo == "BF":
        with open(tsp,"r") as file:
            lines = file.readlines()

        coordinates = []
        for line in lines:
            if line.startswith("EOF"):
                break
            if line[0].isdigit():
                id, x, y = map(float, line.split()[0:])
                coordinates.append((id, x, y))
        best_tour, distance, total_time = brute_force_tsp(coordinates,cutoff)

        print('BF Algo Runtime: ' + str(total_time))

        with open(os.path.join(output_dir, sol_file), 'w') as f:
            f.write(str(int(distance)) + "\n")
            f.write(','.join([str(int(vertex[0])) for vertex in best_tour]))
        f.close()
    
    if algo == 'Approx':
        city_pos = read_tsp_mst(tsp)
        tic = time.time()
        edge_li = find_mst(city_pos)
        best_tour = dfs(edge_li)
        distance = compute_final_cost_mst(best_tour, city_pos)
        total_time = time.time() - tic

        with open(os.path.join(output_dir, sol_file), 'w') as f:
            f.write(str(int(distance)) + "\n")
            f.write(','.join([str(int(vertex[0])) for vertex in best_tour]))
        f.close()

    if algo == "LS":
        start_time = time.time()
        #initialize
        LS_algo = LS(seed, tsp)
        LS_cost, LS_solution = LS_algo.cost_optimization()
        total_time = round((time.time() - start_time), 5)
        
        print('LS Algo Runtime: ' + str(total_time))
        
        sol_file = "_".join([tsp_name, algo,str(seed)]) + '.sol'
        with open(os.path.join(output_dir, sol_file), 'w') as f:
            f.write(str(LS_cost) + "\n")
            f.write(','.join([str(vertex) for vertex in LS_solution]))
        f.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm with specified parameters')
    parser.add_argument('-inst', type=str, required=True, help='graph file')
    parser.add_argument('-alg', default='BF', type=str,
                        required=False, help='Choice Algorithm')
    parser.add_argument('-time', default=600, type=int,
                        required=False, help='Cutoff running time for algorithm')
    parser.add_argument('-seed',type=int, default=0, required=False, help='Random Seed for algorithm')
    args = parser.parse_args()

    main(args.inst, args.alg, args.time, args.seed)
