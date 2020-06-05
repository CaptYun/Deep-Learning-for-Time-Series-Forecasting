## CNNs for Time Series Forecasting   
### Univariate CNN Models   
univariate : a single series of observations with a temporal ordering   
sequence -> input : 3 time steps, output : 1 time step for 1 step prediction     
```python
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

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

raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps = 3
X, y = split_sequence(raw_seq, n_steps)

print('(X, y)')

for i in range(len(X)):
  print(X[i], y[i])

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features)) #[samples,timesteps]->[samples,timesteps,features]

for i in range(len(X)):
  print(X[i], y[i])
```
(X, y)   
[10 20 30] 40    
[20 30 40] 50    
[30 40 50] 60    
[40 50 60] 70    
[50 60 70] 80    
[60 70 80] 90    
[[10]    
 [20]    
 [30]] 40    
[[20]    
 [30]    
 [40]] 50    
[[30]    
 [40]    
 [50]] 60    
[[40]    
 [50]    
 [60]] 70    
[[50]    
 [60]    
 [70]] 80    
[[60]    
 [70]    
 [80]] 90    
 ```python
 model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
x_input = array([70,80,90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[100.82957]]
##   
### Multivariate CNN Models
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

def split_sequences(sequences, n_steps):
  X, y = list(), list()
  for i in range(len(sequences)):
    end_ix = i + n_steps
    if end_ix > len(sequences):
      break
    seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
print(out_seq)

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = np.hstack((in_seq1, in_seq2, out_seq))
print(dataset)

n_steps = 3
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
for i in range(len(X)):
  print(X[i], y[i])

n_features = X.shape[2]
```
[ 25  45  65  85 105 125 145 165 185]   
[[ 10  15  25]   
 [ 20  25  45]   
 [ 30  35  65]   
 [ 40  45  85]   
 [ 50  55 105]   
 [ 60  65 125]   
 [ 70  75 145]   
 [ 80  85 165]   
 [ 90  95 185]]   
(7, 3, 2) (7,)   
[[10 15]   
 [20 25]   
 [30 35]] 65   
[[20 25]   
 [30 35]   
 [40 45]] 85   
[[30 35]   
 [40 45]   
 [50 55]] 105   
[[40 45]   
 [50 55]   
 [60 65]] 125   
[[50 55]   
 [60 65]   
 [70 75]] 145   
[[60 65]   
 [70 75]   
 [80 85]] 165   
[[70 75]   
 [80 85]   
 [90 95]] 185   
 ```python
 model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
x_input = array([[80,85], [90,95], [100,105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
```
[[208.13545]]   
###    
#### Multi-headed CNN Model
각 submodel의 output을 prediction이전에 combine해서 output sequence로 보냄
![Multi-headed](https://user-images.githubusercontent.com/63143652/83852211-6c086a80-a74e-11ea-94ed-5a399ff0e576.jpeg){: width="50" height="50"}
