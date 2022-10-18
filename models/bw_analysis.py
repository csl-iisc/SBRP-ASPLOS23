file_name = "buff_srp_discrete_eager/eager_stats"
file1 = open(file_name)
Lines = file1.readlines()

granularity = 1

count1 = []
sample_value = -1
for line in Lines:
    if line.split()[0] == '0':
        sample_value = sample_value + 1
        count1.append(0)
    count1[sample_value // granularity] = count1[sample_value // granularity] + int(line.split()[1])

file_name = "buff_srp_discrete_lazy/lazy_stats"
file1 = open(file_name)
Lines = file1.readlines()

count2 = []
sample_value = -1
for line in Lines:
    if line.split()[0] == '0':
        sample_value = sample_value + 1
        count2.append(0)
    count2[sample_value // granularity] = count2[sample_value // granularity] + int(line.split()[1])

file_name = "buff_srp_discrete_of6/window_stats"
file1 = open(file_name)
Lines = file1.readlines()

count3 = []
sample_value = -1
for line in Lines:
    if line.split()[0] == '0':
        sample_value = sample_value + 1
        count3.append(0)
    count3[sample_value // granularity] = count3[sample_value // granularity] + int(line.split()[1])

'''
for x in range(max(len(count1), len(count2), len(count3))):
	lazy = 0
	eager = 0
	window = 0
	if x < len(count1):
		lazy = lazy + count1[x]
	if x < len(count2):
		eager = eager + count2[x]
	if x < len(count3):
		window = window + count3[x]
	print(str(x) + "\t" + str(lazy) + "\t" + str(eager) + "\t" + str(window))
'''
samples = 150
i = 0
j = 0
k = 0
print("\tLazy\tEager\tWindow")
for x in range(samples - 1):
	lazy = 0
	eager = 0
	window = 0
	for y in range(len(count1) // samples):
		lazy = lazy + count1[i]
		i = i + 1
	for y in range(len(count2) // samples):
		eager = eager + count2[j]
		j = j + 1
	for y in range(len(count3) // samples):
		window = window + count3[k]
		k = k + 1
	print(str(x) + "\t" + str(lazy) + "\t" + str(eager) + "\t" + str(window))

lazy = 0
eager = 0
window = 0
while i < len(count1):
	lazy = lazy + count1[i]
	i = i + 1
while j < len(count2):
	eager = eager + count2[j]
	j = j + 1
while k < len(count3):
	window = window + count3[k]
	k = k + 1
print(str(samples - 1) + "\t" + str(lazy) + "\t" + str(eager) + "\t" + str(window))
