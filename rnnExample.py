import networkcrafter as nc


trainingSequence = ['t', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k', ' ', 'b', 'r', 'o', 'w', 'n', ' ', 'f', 'o', 'x', ' ', 'j', 'u', 'm', 'p', 's', ' ', 'o', 'v', 'e', 'r', ' ', ' ', 't', 'h', 'e', ' ', ' ', 'l', 'a', 'z', 'y', ' ', ' ', 'd', 'o', 'g']

trainingNums = []
[trainingNums.append(ord(letter)) for letter in trainingSequence]

targets = trainingNums[1:]
targets.append(ord('.'))


print trainingNums
print targets

inLayer = nc.InputLayer(1)
