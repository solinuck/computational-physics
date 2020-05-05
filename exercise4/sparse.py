def multA(x):
    res = np.empty_like(x)
    for i, v in enumerate(val):
        res[row[i]] += v*x[col[i]]
    return res

val = np.array([1, 1, 1, 9, 4])
row = np.array([0, 2, 1, 4, 8])
col = np.array([0, 2, 1, 3, 8])

def print_sparse(val, row, col):
    s = list("0" * (np.max(row)+1) * (np.max(col)+1))

    c = 0
    for i,v in enumerate(val):
        s[row[i] + (np.max(row)+1)*col[i]] = str(v)
    for i in range(1, np.max(col)+1):
        s.insert(i*(np.max(row)+2)-1, "\n")
    print("".join(s))
