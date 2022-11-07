import numpy as np
import matplotlib.pyplot as plt
import os

bar_width = 0.16
fig = plt.subplots(figsize = (20, 12))

benchmarks = ["KVS", "HM", "SRAD", "Red", "MQ", "Scan"]
models = ["epoch_near", "sbrp_near"]

baseline = "epoch_near"
figure11_speedup = []

path = "../figure11_results/"
os.chdir(path)

graph_path = "../outputs/figure11_graph.pdf"
output_file_path = "../outputs/figure11_output.txt"

baseline_latency = {}

# Get baseline recovery percentages for all benchmarks
output_file = open(output_file_path, "w")
for benchmark in benchmarks:
    baseline_latency[benchmark] = 0
    tot_execution_cycles =0
    recovery_content = []
    file_name = "epoch_near_" + benchmark + "_rec.out"
    file_read = open(file_name, "r")
    lines = file_read.readlines()
    print(benchmark)
    for line in lines:
        if "Recovery begins" in line:
            start_idx = lines.index(line)
        if "Recovery ends" in line:
            end_idx = lines.index(line)
            break
    for index in range(start_idx, end_idx):
        if "gpu_sim_cycle" in lines[index]:
            baseline_latency[benchmark] += float(lines[index].split(" ")[-1])
    file_read.close()
    print("baseline_latency: " + str(baseline_latency[benchmark]))


# Get speedup for each model and each benchmark
for model in models:
    speedup_list = []
    speedup_list_str = []
    for benchmark in benchmarks:
        recovery_content = []
        latency = 0
        tot_execution_cycles = 0
        file_name = model + "_" + benchmark + "_rec.out"
        print(file_name)
        file_read = open(file_name, "r")
        lines = file_read.readlines()
        for line in lines:
            if "Recovery begins" in line:
                start_idx = lines.index(line)
            if "Recovery ends" in line:
                end_idx = lines.index(line)
                break
        for index in range(start_idx, end_idx):
            if "gpu_sim_cycle" in lines[index]:
                latency += int(lines[index].split(" ")[-1])
        file_read.close()
        speedup = float(latency)/float(baseline_latency[benchmark])
        print("latency: " + str(latency))
        print(f"model: " + model + " baseline latency: " + str(latency) + "   model_latency: " + str(baseline_latency[benchmark]) + " speedup: " + str(speedup))
        speedup_list.append(speedup)
    figure11_speedup.append(speedup_list)
    file_read.close()

print(speedup_list)

# Write to output file
for speedup in figure11_speedup:
    for i in speedup:
        output_file.write(str(i) + ", ")
    output_file.write("\n")

output_file.close()
print(figure11_speedup)

# Bar for each model
br = []
br.append(np.arange(len(benchmarks)))
for i in range(1, len(models)):
    br.append([x + bar_width for x in br[i-1]])

colors = ['r', 'g', 'b', 'c', 'm']
for i in range(len(models)):
    print(br[i])
    print(i)
    plt.bar(br[i], figure11_speedup[i], color = colors[i], width = bar_width, edgecolor = 'black', label = models[i])

plt.ylabel('Normalized run time', fontweight = 'bold', fontsize = 15)
plt.xticks([r + 2 * bar_width for r in range(len(benchmarks))],
           ['gpKVS', 'HM', 'SRAD', 'Red', 'MQ', 'Scan'])

plt.legend()
plt.savefig(graph_path, format='pdf')
plt.show()
