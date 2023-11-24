import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

root_folder = Path(__file__).parents[0]

def readplot(file, label1, label2):
    my_path = root_folder / ("info\\" + file)
    file1 = open(my_path, 'r')
    Lines = file1.readlines()
    best = []
    avrg = []
    for line in Lines:
        stat = list(map(float, line.split()))
        best.append(stat[0])
        avrg.append(stat[1])
    plt.plot(best, label=label1)
    #plt.plot(avrg, label=label2)

readplot("basestats.txt", "Best (base)", "Average (base)")
readplot("badapplestats.txt", "Best (badapple)", "Average (badapple)")
readplot("test2stats.txt", "Best (test2)", "Average (test2)")
plt.title("Best per generation")
plt.xlabel("Generation")
plt.ylabel("fitness (from 0 to 2)")
plt.legend()
plt.ylim(0, 2)
plt.show()