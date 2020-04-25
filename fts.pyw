import warnings
warnings.filterwarnings('ignore')


import matplotlib.pylab as plt

# pylab inline
from pyFTS.data import Enrollments

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,5])

df = Enrollments.get_dataframe()
plt.plot(df['Year'], df['Enrollments'])

data = df['Enrollments'].values

# plt.show()




from pyFTS.partitioners import Grid

fs = Grid.GridPartitioner(data=data, npart=4)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15, 5])

fs.plot(ax)
# plt.show()

fuzzyfied = fs.fuzzyfy(data, method='maximum', mode='sets')

# print(fuzzyfied)

from pyFTS.common import FLR

patterns = FLR.generate_non_recurrent_flrs(fuzzyfied)

# print([str(k) for k in patterns])
from pyFTS.models import chen


model = chen.ConventionalFTS(partitioner=fs)
model.fit(data)
# print(model)

from pyFTS.common import Util

# Util.plot_rules(model, size=[15, 5], rules_by_axis=10)


fuzzyfied = fs.fuzzyfy(18876, method='maximum', mode='sets')

# print(fuzzyfied)

# print(model.predict([18876]))


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])

forecasts = model.predict(data)
forecasts.insert(0,None)

orig, = plt.plot(data, label="Original data")
pred, = plt.plot(forecasts, label="Forecasts")

plt.legend(handles=[orig, pred])

plt.show()