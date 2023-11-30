import time
import os
from math import sqrt
import argparse
import itertools


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

# 写你们的function：
    #def()

def main(tsp,algo,cutoff):
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

    ##在下面分别写你们的
    # if algo == "Approx":

    # if algo == "LS":


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm with specified parameters')
    parser.add_argument('-inst', type=str, required=True, help='graph file')
    parser.add_argument('-time', default=600, type=int,
                        required=False, help='Cutoff running time for algorithm')
    parser.add_argument('-seed',type=int, required=False, help='Random Seed for algorithm')
    parser.add_argument('-alg', default='BF', type=str,
                        required=False, help='Choice Algorithm')
    args = parser.parse_args()

    main(args.inst, args.alg, args.time)