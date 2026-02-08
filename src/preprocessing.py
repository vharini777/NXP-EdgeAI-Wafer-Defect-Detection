import tensorflow as tf

def get_augmentation_pipeline():
    """
    Returns the data augmentation sequence optimized for 
    wafer defect detection.
    """
    return tf.keras.Sequential([
        # Replacing RandomResizedCrop with Zoom for compatibility
        tf.keras.layers.RandomZoom(height_factor=(-0.3, -0.1), fill_mode='reflect'),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentation")

def preprocess_image(image, label):
    """
    Standardizes image size and pixel range for MobileNetV3 input.
    """
    image = tf.image.resize(image, (224, 224))
    # MobileNetV3 typically expects [0, 255] for internal scaling, 
    # but 1.0/255 normalization is standard for custom training.
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label