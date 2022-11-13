import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import approximation as approx

def normalize_data(path):
    with open(path) as file_handler:
        inputs = []
        n_inputs = []
        outputs = []
        n_outputs = []
        pins = []
        n_pins = []
        diarea = []

        inArea = False

        for i in file_handler:
            line = i.strip()
            vector = line.split()
            if len(vector) == 0:
                continue
            if vector[0] == "DIEAERA":
                inArea = True
                diarea.append(((int(vector[2]), int(vector[3])), (int(vector[6]), int(vector[7]))))
            elif vector[-1] == ';':
                inArea = False

            if inArea and vector[0] != "DIEAERA":
                diarea.append(((int(vector[1]), int(vector[2])), (int(vector[5]), int(vector[6]))))

            if len(vector) > 1 and vector[0] == '-':
                if len(inputs) < 16:
                    n_inputs.append(vector[1])
                else:
                    n_outputs.append(vector[1])

            if len(vector) > 1 and vector[1] == 'FIX':
                if len(inputs) < 16:
                    inputs.append((0, int(vector[4])))
                else:
                    outputs.append((0, int(vector[4])))

            if line[0] == "i":
                pins.append(((int(vector[5])), int(vector[6])))
                n_pins.append(vector[0])

        return pins, inputs, outputs, diarea, n_inputs, n_outputs, n_pins

def check_in_area(x, y, area) -> bool:
    if x >= area[0][0] and x <= area[1][0] and y >= area[0][1] and y <= area[1][1]:
        return True
    else:
        return False

def get_section_points(x_pins, y_pins, diarea) -> tuple[int, int, int]:
    seccions = []
    x_sec = []
    y_sec = []

    N_SCALE = 40
    x_inc = x_max//N_SCALE
    y_inc = y_max//N_SCALE

    for i in range(N_SCALE+1):
        for j in range(N_SCALE+1):
            for area in diarea:
                x = i*x_inc
                y = j*y_inc
                if check_in_area(x, y, area):
                    seccions.append((x, y))
                    x_sec.append(x)
                    y_sec.append(y)
                    break
    return seccions, x_sec, y_sec

def seccionar_area(seccions, x_max, y_max) -> tuple[int, int]:
    y_25 = (y_max//100)*25
    y_75 = (y_max//100)*75
    x_50 = (x_max//100)*50
    x_75 = (x_max//100)*75
    sectors = [[], [], [], [], []]
    for seccio in seccions:
        if seccio[0] <= x_50 and seccio[1] <= y_25:
            sectors[0].append(seccio)
        elif seccio[0] >= x_50  and seccio[1] <= y_25:
            sectors[1].append(seccio)
        elif seccio[0] >= x_75 and seccio[1] >= y_25 and seccio[1] <= y_75:
            sectors[2].append(seccio)
        elif seccio[0] <= x_50 and seccio[1] >= y_75:
            sectors[3].append(seccio)
        elif seccio[0] >= x_50 and seccio[1] >= y_75:
            sectors[4].append(seccio)
    n_sectors = [len(sector) for sector in sectors]
    return sectors, n_sectors

def calculate_drivers(n_sectors, drivers) -> int:
    number = sum(n_sectors)
    n_drivers = []
    n_drivers_decimal = []
    for i in n_sectors:
        num = drivers*i/number
        n_drivers.append(int(num))
        n_drivers_decimal.append(num - int(num))
    while sum(n_drivers) < drivers and 0 in n_drivers:
        for i in range(len(n_drivers)):
            if n_drivers[i] == 0:
                n_drivers[i] += 1
                break
    if sum(n_drivers) < drivers:
        n_drivers[2] += 1
    if sum(n_drivers) < drivers - 1:

        n_drivers[0] += 1
        n_drivers[1] += 1
    while sum(n_drivers) < drivers:
        ind = n_drivers_decimal.index(max(n_drivers_decimal))
        n_drivers_decimal[ind] = 0
        n_drivers[ind] += 1

    return n_drivers

def calculate_separators(n_drivers) -> int:
    y_25 = (y_max//100)*25
    y_50 = (y_max//100)*50
    y_75 = (y_max//100)*75
    x_50 = (x_max//100)*50
    x_75 = (x_max//100)*75
    separators = []

    for i in range(1, n_drivers[3]):
        x_sep = x_50//(n_drivers[3])
        separators.append((x_sep*i, y_75))
    separators.append((x_50, y_75))

    for i in range(1, n_drivers[4]+1):
        x_sep = x_50//(n_drivers[4]+1)
        separators.append((x_50 + x_sep*i, y_75))
    #separators.append((x_max, y_75))

    for i in range(1, n_drivers[2]+1):
        y_sep = y_50//(n_drivers[2]+1)
        separators.append((x_75, y_75-y_sep*i))
    #separators.append((x_max, y_25))

    for i in range(1, n_drivers[1]):
        x_sep = x_50//(n_drivers[1])
        separators.append((x_max - x_sep*i, y_25))
    separators.append((x_50, y_25))

    for i in range(1, n_drivers[0]):
        x_sep = x_50//(n_drivers[0])
        separators.append((x_50 - x_sep*i, y_25))
    return separators

def trace_lines(separators) -> int:
    line_equations = []
    for i in range(len(separators)):
        a = (separators[i][1] - y_max/2)/separators[i][0]
        b = y_max//2
        line_equations.append((a, b))
    return line_equations


def cluster_pins(pins, line_equations) -> int:
    clustered_pins = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    for pin in pins:
        if pin[1] >= line_equations[0][0]*pin[0] + line_equations[0][1]:
            clustered_pins[0].append(pin)

        elif pin[1] >= line_equations[1][0]*pin[0] + line_equations[1][1]:
            clustered_pins[1].append(pin)
        elif pin[1] >= line_equations[2][0]*pin[0] + line_equations[2][1]:
            clustered_pins[2].append(pin)
        elif pin[1] >= line_equations[3][0]*pin[0] + line_equations[3][1]:
            clustered_pins[3].append(pin)
        elif pin[1] >= line_equations[4][0]*pin[0] + line_equations[4][1]:
            clustered_pins[4].append(pin)
        elif pin[1] >= line_equations[5][0]*pin[0] + line_equations[5][1]:
            clustered_pins[5].append(pin)
        elif pin[1] >= line_equations[6][0]*pin[0] + line_equations[6][1]:
            clustered_pins[6].append(pin)
        elif pin[1] >= line_equations[7][0]*pin[0] + line_equations[7][1]:
            clustered_pins[7].append(pin)
        elif pin[1] >= line_equations[8][0]*pin[0] + line_equations[8][1]:
            clustered_pins[8].append(pin)
        elif pin[1] >= line_equations[9][0]*pin[0] + line_equations[9][1]:
            clustered_pins[9].append(pin)
        elif pin[1] >= line_equations[10][0]*pin[0] + line_equations[10][1]:
            clustered_pins[10].append(pin)
        elif pin[1] >= line_equations[11][0]*pin[0] + line_equations[11][1]:
            clustered_pins[11].append(pin)
        elif pin[1] >= line_equations[12][0]*pin[0] + line_equations[12][1]:
            clustered_pins[12].append(pin)
        elif pin[1] >= line_equations[13][0]*pin[0] + line_equations[13][1]:
            clustered_pins[13].append(pin)
        elif pin[1] >= line_equations[14][0]*pin[0] + line_equations[14][1]:
            clustered_pins[14].append(pin)
        else:
            clustered_pins[15].append(pin)
    return clustered_pins


def get_pin_name(pin, pins, n_pins):
    #print(pin)
    #print(pins.index(pin))
    #print(n_pins[pins.index(pin)])
    return n_pins[pins.index(pin)]


def write_output(cycles, inputs, n_inputs, outputs, n_outputs, pins, n_pins, clustered_pins):
    clustered_pins = clustered_pins[::-1]
    with open(path_out, 'w') as output_handler:
        for j in range(len(cycles)):
            #print(cycles[j])
            #print(clustered_pins[j])
            for i in range(len(cycles[j])-1):
                output_handler.write("- NET: " + str(j) + '\n')
                if i == 0:
                    name_i = n_inputs[j]
                    output_handler.write('( ' + name_i + ' conn_in )\n')
                    pin1 = clustered_pins[j][cycles[j][i+1]-1]
                    name_p1 = get_pin_name(pin1, pins, n_pins)
                    output_handler.write('( ' + name_p1 + ' conn_out )\n')
                elif i == len(cycles[j])-2:
                    pin = clustered_pins[j][cycles[j][i]-1]
                    name_p1 = get_pin_name(pin, pins, n_pins)
                    output_handler.write('( ' + name_p1 + ' conn_in )\n')
                    name_o = n_outputs[j]
                    output_handler.write('( ' + name_o + ' conn_out )\n')
                else:
                    #print(cycles[j][i]-1)
                    #print(clustered_pins[j])
                    pin1 = clustered_pins[j][cycles[j][i]-1]
                    name_p1 = get_pin_name(pin1, pins, n_pins)
                    output_handler.write('( ' + name_p1 + ' conn_in )\n')
                    pin2 = clustered_pins[j][cycles[j][i+1]-1]
                    name_p2 = get_pin_name(pin2, pins, n_pins)
                    output_handler.write('( ' + name_p2 + ' conn_out )\n')
                output_handler.write(";\n")


def scatter_pins(x_pins, y_pins):
    plt.scatter(x_pins, y_pins, s=0.02)


def scatter_drivers(x_inps, x_outs, y_inps, y_outs):
    plt.scatter(x_inps, y_inps, s=2, c='g')
    plt.scatter(x_outs, y_outs, s=2, c='r')


def scatter_seccions(x_sec, y_sec):
    plt.scatter(x_sec, y_sec, c='y', s=3)


def scatter_sectors(sectors):
    colors =['orange', 'pink', 'purple', 'cyan', 'yellow']
    for color in colors:
        for i, sector in enumerate(sectors):
            x_es = [p[0] for p in sector]
            y_es = [p[1] for p in sector]
            plt.scatter(x_es, y_es, s=3, c=colors[i])


def scatter_clustered_pins(clustered_pins):
    for i in range(len(clustered_pins)):
        x_cpins = [pin[0] for pin in clustered_pins[i]]
        y_cpins = [pin[1] for pin in clustered_pins[i]]
        plt.scatter(x_cpins, y_cpins, s=0.02)


def scatter_separators(separators):
    x_sep = [s[0] for s in separators]
    y_sep = [s[1] for s in separators]
    plt.scatter(x_sep, y_sep, s=5, c='black')


def scatter_line_equations(line_equations):
    for i in range(len(line_equations)):
        x_line = np.linspace(0, x_max, num=100)
        y_line = []
        for j in range(len(x_line)):
            v = line_equations[i][0]*x_line[j] + line_equations[i][1]
            if v < 0:
                x_line = x_line[:j]
                break
            elif v > y_max:
                x_line = x_line[:j]
                break
            y_line.append(v)

        plt.plot(x_line, y_line)


def sim_annealing(points):
    n = len(points)
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            if (i,j)==(0,n-1):
                G.add_edge(i, j, weight= 10_000_000_000)
            elif (i, j)==(n-1, 0):
                G.add_edge(i, j, weight = -10_000_000_000)
            else:
                dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                G.add_edge(i, j, weight = dist)
    cycle = approx.simulated_annealing_tsp(G, "greedy")

    ys1 = [points[point][1] for point in cycle]
    xs1 = [points[point][0] for point in cycle]
    plt.plot(xs1, ys1)
    #plt.show()
    #print(cycle)
    if cycle[1] == n-1:
        cycle = cycle[1:][::-1]
    else:
        cycle = cycle[:-1]
    #print(points)
    #print(cycle)
    return cycle



if __name__ == "__main__":
    path_in = "testcase.def"
    path_out = "output.def"
    DRIVERS = 16

    pins, inputs, outputs, diarea, n_inputs, n_outputs, n_pins = normalize_data(path_in)

    x_pins = [pin[0] for pin in pins]
    y_pins = [pin[1] for pin in pins]
    x_inps = [inp[0] for inp in inputs]
    y_inps = [inp[1] for inp in inputs]
    x_outs = [out[0] for out in outputs]
    y_outs = [out[1] for out in outputs]

    x_max = max(x_pins)
    y_max = max(y_pins)

    scatter_pins(x_pins, y_pins)
    scatter_drivers(x_inps, x_outs, y_inps, y_outs)

    seccions, x_sec, y_sec = get_section_points(x_pins, y_pins, diarea)
    #scatter_seccions(x_sec, y_sec)

    sectors, n_sectors = seccionar_area(seccions, x_max, y_max)
    #scatter_sectors(sectors)

    n_drivers = calculate_drivers(n_sectors, DRIVERS)
    separators = calculate_separators(n_drivers)
    #scatter_separators(separators)

    line_equations = trace_lines(separators)
    clustered_pins = cluster_pins(pins, line_equations)

    cycles = []
    distances = []
    for i in range(16):
        all_pins = [(x_inps[i], y_inps[i])] + clustered_pins[15-i] + [(x_outs[i], y_outs[i])]
        #print(len(all_pins), len(clustered_pins[15-i]))
        cycle = sim_annealing(all_pins)
        cycles.append(cycle)
        dist = 0
        for j in range(len(all_pins)-1):
            dist += abs(all_pins[j][0] - all_pins[j+1][0]) + abs(all_pins[j][1] - all_pins[j+1][1])
        distances.append(dist)


    write_output(cycles, inputs, n_inputs, outputs, n_outputs, pins, n_pins, clustered_pins)

    plt.show()
