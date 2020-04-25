import pandas as pd
import numpy as np

df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [77, 78, 79]]),
                   columns=['a', 'b', 'c'])

print(df2)
print(df2.columns)