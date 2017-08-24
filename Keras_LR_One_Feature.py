import numpy as np
from keras.models import Sequential 
from keras.layers import Dense 
import matplotlib.pyplot as plt 
print ("Import finished")

X = np.linspace(0, 2, 300) 
np.random.shuffle(X)
Y = 3 * X + np.random.randn(*X.shape) * 0.33

plt.scatter(X,Y)
plt.show()
print (X[:10],'\n',Y[:10])

X_train,Y_train = X[:260],Y[:260]
X_test,Y_test = X[260:],Y[260:]

model = Sequential()
model.add(Dense(units=1, kernel_initializer="uniform", activation="linear", input_dim=1))
weights = model.layers[0].get_weights() 
w_init = weights[0][0][0] 
b_init = weights[1][0] 
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) 

model.compile(loss='mse', optimizer='sgd')

model.fit(X_train, Y_train, epochs=500, verbose=1)

Y_pred = model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()
weights = model.layers[0].get_weights() 
w_init = weights[0][0][0] 
b_init = weights[1][0] 
print('Linear regression model is trained with weights w: %.2f, b: %.2f' % (w_init, b_init)) 

a = np.array([1.66])
Pre=model.predict(a)
print (Pre)





