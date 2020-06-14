import pickle

def pickle_it(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)




def unpickle_it(path):
    with open(path, 'rb') as f:
        return pickle.load(f)