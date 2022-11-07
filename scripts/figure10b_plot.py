import numpy as np
import matplotlib.pyplot as plt
import os

bar_width = 0.25
fig = plt.subplots(figsize = (20, 12))

benchmarks = ["KVS", "HM", "SRAD", "Red", "MQ", "Scan"]
models = ["sbrp_near_bw_half", "sbrp_near", "sbrp_near_bw_double"]
baseline_models = ["epoch_near_bw_half", "epoch_near" , "epoch_near_bw_double"]

baseline = "epoch_near"
figure10_b_speedup = []

path = "../figure10_b_results/"
os.chdir(path)

graph_path = "../outputs/figure10_b_graph.pdf"
output_file_path = "../outputs/figure10_b_output.txt"

baseline_latency = {}

# Get baseline latency for all benchmarks
output_file = open(output_file_path, "w")

# Get speedup for each model and each benchmark
for model in models:
    speedup_list = []
    speedup_list_str = []
    for benchmark in benchmarks:
        baseline_latency = 0
        file_name = baseline_models[models.index(model)] + "_" + benchmark + ".out"
        print(file_name)
        file_read = open(file_name, "r")
        lines = file_read.readlines()
        for line in reversed(lines):
            if "gpu_tot_sim_cycle" in line:
                baseline_latency = line.split(" ")[-1]
                break
        file_read.close()
        file_name = model + "_" + benchmark + ".out"
        print(file_name)
        file_read = open(file_name, "r")
        lines = file_read.readlines()
        for line in reversed(lines):
            if "gpu_tot_sim_cycle" in line:
                latency = line.split(" ")[-1]
                speedup = float(baseline_latency)/float(latency)
                print(f"model: " + model + " baseline latency: " + baseline_latency + " model_latency: " + latency + " speedup: " + str(speedup))
                speedup_list.append(speedup)
                break
    figure10_b_speedup.append(speedup_list)
    file_read.close()

print(speedup_list)

# Write to output file
output_file.write(" , ")
for benchmark in benchmarks:
    output_file.write(benchmark + ", ")
output_file.write("\n")
for speedup in figure10_b_speedup:
    output_file.write(models[figure10_b_speedup.index(speedup)] + ", ")
    for i in speedup:
        output_file.write(str(i) + ", ")
    output_file.write("\n")

output_file.close()
print(figure10_b_speedup)

# Bar for each model
br = []
br.append(np.arange(len(benchmarks)))
for i in range(1, len(models)):
    br.append([x + bar_width for x in br[i-1]])

colors = ['r', 'g', 'b', 'c', 'm']
for i in range(len(models)):
    print(br[i])
    print(i)
    plt.bar(br[i], figure10_b_speedup[i], color = colors[i], width = bar_width, edgecolor = 'black', label = models[i])

plt.ylabel('Normalized run time', fontweight = 'bold', fontsize = 15)
plt.xticks([r + 1.5 * bar_width for r in range(len(benchmarks))],
           ['gpKVS', 'HM', 'SRAD', 'Red', 'MQ', 'Scan'])

plt.legend()
plt.savefig(graph_path, format='pdf')
plt.show()
