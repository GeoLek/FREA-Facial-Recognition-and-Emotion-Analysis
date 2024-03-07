import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Dataset paths
base_dir = '/Work-data'
train_dir = f'{base_dir}/train'
validation_dir = f'{base_dir}/validation'

# Data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Note: Adjust the batch size if necessary depending on your GPU memory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),  # EfficientNetB7 expects 300x300 input
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),  # Same target size for validation
    batch_size=32,
    class_mode='categorical'
)

# Model setup
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback for saving models
model_checkpoint = ModelCheckpoint(
    'efficientnetb7_model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    mode='min'
)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,  # Adjust epochs based on your dataset and early stopping criteria
    validation_data=validation_generator,
    callbacks=[model_checkpoint]
)

# Plotting and saving the validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss Over Epochs - EfficientNetB7')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('efficientnetb7_validation_loss_plot.png')
plt.show()
