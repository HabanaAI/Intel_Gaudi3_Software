def calc_ratio(val1, val2):
    return float(((val2 - val1) / val1) * 100.0) if val1 and val2 else 0

def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]
