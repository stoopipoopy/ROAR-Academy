import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Create synthetic dataset
blue_class = np.concatenate([
    np.random.randn(25, 2) * 1 + [-5, 5],
    np.random.randn(25, 2) * 1 + [5, -5]
], axis=0)

red_class = np.concatenate([
    np.random.randn(25, 2) * 1 + [-5, -5],
    np.random.randn(25, 2) * 1 + [5, 5]
], axis=0)

X = np.vstack([red_class, blue_class])
y = np.array([[1, 0]] * 50 + [[0, 1]] * 50)

# Define the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(X, y, epochs=100, batch_size=16, verbose=0)

# Evaluate
loss, acc = model.evaluate(X, y, verbose=0)
print(f"Training Accuracy: {acc * 100:.2f}%")

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid, verbose=0)
    Z = np.argmax(preds, axis=1).reshape(xx.shape)

    plt.scatter(X[:,0], X[:,1], c=np.argmax(y, axis=1), cmap='coolwarm', edgecolors='k')
    plt.title("Decision Boundary of MLP (Keras)")
    plt.show()

plot_decision_boundary(model, X, y)
