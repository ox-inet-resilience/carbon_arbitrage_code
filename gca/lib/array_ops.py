import pandas as pd


def maybe_round6(do_it, x):
    return round(x, 6) if do_it else x


def sum_array_of_mixed_objs(x):
    out = 0.0
    for e in x:
        if isinstance(e, float):
            out += e
        elif isinstance(e, dict):
            out += sum(e.values())
        else:
            out += e.sum()
    return out


def divide_array_of_mixed_objs(arr, divider):
    out = []
    for e in arr:
        if isinstance(e, dict):
            out.append({k: v / divider for k, v in e.items()})
        else:
            # float or Pandas Series
            out.append(e / divider)
    return out


def add_array_of_mixed_objs(x, y):
    assert len(x) == len(y)
    out = []
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if isinstance(xi, pd.Series):
            xi = xi.to_dict()
        if isinstance(yi, pd.Series):
            yi = yi.to_dict()

        if isinstance(xi, dict):
            if isinstance(yi, float) and yi == 0.0:
                out.append(xi.copy())
                continue
            z = {}
            # We need to include keys from both xi and yi, because recently in
            # the coal export, there are 100% importer countries that are not
            # part of masterdata.
            for key in set(xi) | set(yi):
                z[key] = xi.get(key, 0) + yi.get(key, 0)
            out.append(z)
        else:
            # float
            out.append(xi + yi)
    return out
