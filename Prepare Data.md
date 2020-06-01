## How to Prepare Time Series Data for CNNs and LSTMs    
##   
### 3D Data Preparation Basics   
첫번째 layer : input layer   

3D input data : [samples, timesteps, features], CNNs, LSTMs   

2D input data : [samples, features], univariate time series   
#### # Transform univariate 2d to 3d
```python
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
  X, y = list(), list()
  for i in range(len(sequence)):
    end_ix = i + n_steps
    if end_ix >= len(sequence):
      break
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

series = array([1,2,3,4,5,6,7,8,9,10])
print(series.shape)

# transform to a supervised learning problem
X, y = split_sequence(series, 3)
print(X.shape, y.shape)

# transform input [samples,features] to [samples,timesteps,features]
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X.shape)
```
(10,)
(7, 3) (7,)
(7, 3, 1)

