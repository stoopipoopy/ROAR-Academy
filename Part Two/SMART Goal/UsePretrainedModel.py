import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import kagglehub

# Download latest version
path = kagglehub.dataset_download("xainano/handwrittenmathsymbols")
print("Downloaded to:", path)
print(os.listdir(path))
print("Path to dataset files:", path)




img_size = 64
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    path,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    path,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
'''
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2, restore_best_weights = False)
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[early]
)
'''

model = tf.keras.models.load_model("math_symbol_recognition_model.keras")


loss, acc = model.evaluate(val_data)
print(f"Validation accuracy: {acc * 100:.2f}%")

img = image.load_img('/Users/petropapahadjopoulos/ROAR-Academy/Part Two/SMART Goal/sample_!.png', target_size=(img_size, img_size), color_mode='grayscale')
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
label_map = train_data.class_indices
label_map_inv = {v: k for k, v in label_map.items()}
print(f"Predicted symbol: {label_map_inv[predicted_class]}")
model.save("math_symbol_recognition_model.keras")