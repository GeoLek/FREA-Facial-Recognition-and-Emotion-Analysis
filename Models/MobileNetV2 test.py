import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Corrected: Load the trained model directly into 'model'
model = tf.keras.models.load_model('/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/mobile_net_model_epoch_07_val_loss_1.36.h5')

# Dataset paths
base_dir = '/Work-data'
test_dir = f'{base_dir}/test'

# Test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for correct label ordering
)

# Predict the output on the test data
predictions = model.predict(test_generator, steps=np.ceil(test_generator.samples/test_generator.batch_size))

# Get the index of the maximum value for each prediction
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes

# Class labels (ensure these are in the same order as the training labels)
class_labels = list(test_generator.class_indices.keys())

# Display classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Save the classification report to a CSV file
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv', index=True)

# Compute and plot confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()