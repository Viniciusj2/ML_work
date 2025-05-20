from numpy import load

data = load('raw_positions.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])
