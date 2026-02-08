import tensorflow as tf
from tensorflow.keras import layers

class SpatialAttentionModule(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionModule, self).__init__(**kwargs)
        # 7x7 kernel is standard for spatial attention to capture enough context
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        self.concat = layers.Concatenate(axis=-1)
        self.multiply = layers.Multiply()

    def call(self, input_feature):
        # Generates a mask that "turns off" boring background pixels
        # 1. Spatial pooling
        avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
        
        # 2. Channel concatenation
        concat = self.concat([avg_pool, max_pool])
        
        # 3. Spatial attention map generation
        attention = self.conv(concat)
        
        # 4. Feature refinement
        return self.multiply([input_feature, attention])

    def get_config(self):
        """Allows the model to be saved and reloaded with the custom layer."""
        config = super(SpatialAttentionModule, self).get_config()
        return config