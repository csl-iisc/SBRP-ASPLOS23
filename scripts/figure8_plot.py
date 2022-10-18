import numpy as np
import matplotlib.pyplot as plt
import os

bar_width = 0.16
fig = plt.subplots(figsize = (20, 12))

benchmarks = ["KVS", "HM", "SRAD", "Red", "MQ", "Scan"]
models = ["epoch_far", "sbrp_far", "epoch_near","sbrp_near"]

baseline = "epoch_far"
figure8_rmiss = []

path = "../figure6_results/"
os.chdir(path)

graph_path = "../outputs/figure8_graph.pdf"
output_file_path = "../outputs/figure8_output.txt"

baseline_rmiss = {}

# Get baseline latency for all benchmarks
output_file = open(output_file_path, "w")
for benchmark in benchmarks:
    file_name = "epoch_far_" + benchmark + ".out"
    file_read = open(file_name, "r")
    lines = file_read.readlines()
    for line in reversed(lines):
        if "L1D_total_pm_write_cache_misses" in line:
            baseline_rmiss[benchmark] = line.split(" ")[-1]
            break
file_read.close()


# Get speedup for each model and each benchmark
for model in models:
    rmiss_list = []
    for benchmark in benchmarks:
        file_name = model + "_" + benchmark + ".out"
        file_read = open(file_name, "r")
        lines = file_read.readlines()
        for line in reversed(lines):
            if "L1D_total_pm_read_cache_misses" in line:
                rmiss = line.split(" ")[-1]
                norm_rmiss= (float(rmiss)/float(baseline_rmiss[benchmark])) * 100
                print(benchmark)
                print(f"model: " + model + " baseline rmiss: " + baseline_rmiss[benchmark] + " model_rmiss: " + rmiss + " norm rmiss: " + str(norm_rmiss))
                rmiss_list.append(norm_rmiss)
                break
    figure8_rmiss.append(rmiss_list)
    file_read.close()

print(rmiss_list)

# Write to output file
output_file.write(" , ")
for benchmark in benchmarks:
    output_file.write(benchmark + ", ")
output_file.write("\n")

for rmiss in figure8_rmiss:
    output_file.write(models[figure8_rmiss.index(rmiss)] + ", ")
    for i in rmiss:
        output_file.write(str(i) + ", ")
    output_file.write("\n")

output_file.close()
print(figure8_rmiss)

# Bar for each model
br = []
br.append(np.arange(len(benchmarks)))
for i in range(1, len(models)):
    br.append([x + bar_width for x in br[i-1]])

colors = ['r', 'g', 'b', 'c', 'm']
for i in range(len(models)):
    print(br[i])
    print(i)
    plt.bar(br[i], figure8_rmiss[i], color = colors[i], width = bar_width, edgecolor = 'black', label = models[i])

plt.ylabel('Normalized L1 misses', fontweight = 'bold', fontsize = 20)
plt.xticks([r + 2 * bar_width for r in range(len(benchmarks))],
           ['GPM', 'HM', 'SRAD', 'Red', 'MQ', 'Scan'])

plt.legend()
plt.savefig(graph_path, format='pdf')
plt.show()
