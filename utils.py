import torch


def drop(_X, _Y, _seq_len):
    if _X[-1].size(0) != _seq_len:
        _X, _Y = _X[:-1], _Y[:-1]
    return _X, _Y


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len):
    # split into sequences
    _X = torch.split(_X, _seq_len, dim=0)
    _Y = torch.split(_Y, _seq_len, dim=0)

    _X_test = torch.split(_X_test, _seq_len, dim=0)
    _Y_test = torch.split(_Y_test, _seq_len, dim=0)

    _X, _Y = drop(_X, _Y, _seq_len)
    _X_test, _Y_test = drop(_X_test, _Y_test, _seq_len)

    return _X, _Y, _X_test, _Y_test



def preprocess_data_1(_X, _Y, _seq_len):
    # split into sequences
    _X = torch.split(_X, _seq_len, dim=0)
    _Y = torch.split(_Y, _seq_len, dim=0)


    _X, _Y = drop(_X, _Y, _seq_len)

    return _X, _Y
