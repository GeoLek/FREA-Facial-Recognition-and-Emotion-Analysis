import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Dataset paths
base_dir = '/Work-data'
train_dir = f'{base_dir}/train'
validation_dir = f'{base_dir}/validation'

# Data generators for grayscale images
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    color_mode='grayscale',  # Load images as grayscale
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=64,
    color_mode='grayscale',  # Load images as grayscale
    class_mode='categorical'
)

# Minimal 2D CNN Model setup for grayscale images
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 1)),  # Change to 1 channel for grayscale
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
model_checkpoint = ModelCheckpoint(
    'minimal_cnn_model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_lr=1e-8
)

# Train the model with the new callback
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[model_checkpoint, early_stopping, reduce_lr]
)

# Plotting and saving both training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('minimal_cnn_train_val_loss_plot.png')
plt.show()
