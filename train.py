import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths and parameters
DATASET_DIR = 'D:\multi class'
IMAGE_SIZE = (224, 224) # Example size, adjust as needed
BATCH_SIZE = 32

# Create an ImageDataGenerator for training with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Rescale pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2       # Split data into 80% training, 20% validation
)

# Create a generator for training data
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'          # Specify this is the training set
)

# Create a generator for validation data (no augmentation, just rescaling)
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'        # Specify this is the validation set
)

# Get the number of classes
num_classes = len(train_generator.class_indices)
print(f"Found {train_generator.samples} images belonging to {num_classes} classes for training.")
print(f"Found {validation_generator.samples} images for validation.")