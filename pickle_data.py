import pickle

def pickle_it(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(data, f)

    # f = open(path, "wb")
    #     p = pickle.Pickler(f)
    #     p.dump(data)
        # f.close()

def unpickle_it(path):
    with open(path, 'rb') as f:
        return pickle.load(f)