# ==============================
# Traffic Sign Classification with CNN (Improved, GTSRB, Colab-ready)
# ==============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
to_categorical = tf.keras.utils.to_categorical

# -------------------------
# 1. Load Dataset (Pickle from Drive)
# -------------------------
data_path = "/content/drive/MyDrive/GTSRB/data1.pickle"
save_path = "/content/drive/MyDrive/GTSRB/traffic_sign_model.h5"

with open(data_path, "rb") as f:
    dataset = pickle.load(f, encoding="latin1")

X_train = dataset["x_train"]
y_train = dataset["y_train"]
X_val   = dataset["x_validation"]
y_val   = dataset["y_validation"]
X_test  = dataset["x_test"]
y_test  = dataset["y_test"]

print("Original shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# -------------------------
# Fix channel ordering (N,3,32,32 → N,32,32,3)
# -------------------------
X_train = X_train.transpose(0, 2, 3, 1)
X_val   = X_val.transpose(0, 2, 3, 1)
X_test  = X_test.transpose(0, 2, 3, 1)

print("Fixed shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# Normalize
if X_train.max() > 1.0:
    X_train = X_train / 255.0
    X_val   = X_val / 255.0
    X_test  = X_test / 255.0

# One-hot encode labels
num_classes = 43
y_train = to_categorical(y_train, num_classes)
y_val   = to_categorical(y_val, num_classes)
y_test  = to_categorical(y_test, num_classes)

# -------------------------
# 2. Build Improved CNN Model
# -------------------------
model = models.Sequential([
    layers.Input(shape=(32,32,3)),

    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------
# 3. Data Augmentation
# -------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# -------------------------
# 4. Train Model (with Augmentation)
# -------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=25,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train)//64
)

# -------------------------
# 5. Evaluate Model
# -------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("✅ Test accuracy:", test_acc)

# Plot accuracy & loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.show()

# -------------------------
# 6. Confusion Matrix
# -------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_true, y_pred_classes))

# -------------------------
# 7. Save Model
# -------------------------
model.save(save_path)
print(f"✅ Model saved to: {save_path}")
