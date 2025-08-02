import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

blue = np.concatenate([
    np.random.randn(25, 2) + [-5, -5],
    np.random.randn(25, 2) + [5, 5]
])
red = np.concatenate([
    np.random.randn(25, 2) + [-5, 5],
    np.random.randn(25, 2) + [5, -5]
])
X = np.vstack([blue, red])
y = np.array([[1, 0]] * 50 + [[0, 1]] * 50)

X_transformed = np.column_stack([
    X[:, 0] * X[:, 1], 
    X[:, 0] + X[:, 1] 
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_transformed, y, epochs=100, verbose=0)

loss, acc = model.evaluate(X_transformed, y, verbose=0)
print(f"Accuracy: {acc*100:.2f}%")

def plot_decision_boundary(model, X, y, transform_fn):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_transformed = transform_fn(grid)
    preds = model.predict(grid_transformed, verbose=0)
    Z = np.argmax(preds, axis=1).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:,0], X[:,1], c=np.argmax(y, axis=1), cmap='coolwarm', edgecolor='k')
    plt.title("XOR Decision Boundary (Transformed Features)")
    plt.show()

transform_fn = lambda X: np.column_stack([X[:,0]*X[:,1], X[:,0]+X[:,1]])
plot_decision_boundary(model, X, y, transform_fn)
