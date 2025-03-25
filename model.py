import tensorflow as tf
import numpy as np
import data_utils
import json
import sys

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model

tf.get_logger().setLevel('ERROR')  # Suppress warning messages

# --- Latent space utils --- #
def sample_Z(batch_size, seq_length, latent_dim, use_time=False):
    """
    Generate random samples from latent space.
    
    Args:
        batch_size: Number of samples to generate
        seq_length: Length of time series sequence
        latent_dim: Dimension of latent space
        use_time: If True, sets first dimension to time signal
        
    Returns:
        Tensor of shape [batch_size, seq_length, latent_dim]
    """
    sample = tf.random.normal([batch_size, seq_length, latent_dim], dtype=tf.float32)
    if use_time:
        sample = tf.Variable(sample)
        sample[:, :, 0].assign(tf.linspace(0.0, 1.0, seq_length))
    return sample

# --- Generator model --- #
class Generator(Model):
    """Generator model for GAN architecture using LSTM."""
    
    def __init__(self, hidden_units_g, num_signals):
        """
        Initialize Generator model.
        
        Args:
            hidden_units_g: Number of hidden units in LSTM
            num_signals: Number of output signals (features)
        """
        super(Generator, self).__init__()
        self.lstm = LSTM(hidden_units_g, return_sequences=True)
        self.output_layer = Dense(num_signals, activation='tanh')
        
    def call(self, inputs, training=False):
        """Forward pass through Generator."""
        x = self.lstm(inputs)
        return self.output_layer(x)

# --- Discriminator model --- #
class Discriminator(Model):
    """Discriminator model for GAN architecture using LSTM."""
    
    def __init__(self, hidden_units_d):
        """
        Initialize Discriminator model.
        
        Args:
            hidden_units_d: Number of hidden units in LSTM
        """
        super(Discriminator, self).__init__()
        self.lstm = LSTM(hidden_units_d, return_sequences=True)
        self.output_layer = Dense(1)  # No activation, will be applied in loss
        
    def call(self, inputs, training=False):
        """Forward pass through Discriminator."""
        x = self.lstm(inputs)
        return self.output_layer(x)

# --- Loss functions --- #
def discriminator_loss(real_output, fake_output):
    """
    Calculate discriminator loss.
    
    Args:
        real_output: Discriminator predictions on real data
        fake_output: Discriminator predictions on generated data
        
    Returns:
        Total discriminator loss
    """
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_output, labels=tf.ones_like(real_output)
        )
    )
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_output, labels=tf.zeros_like(fake_output)
        )
    )
    return real_loss + fake_loss

def generator_loss(fake_output):
    """
    Calculate generator loss.
    
    Args:
        fake_output: Discriminator predictions on generated data
        
    Returns:
        Generator loss
    """
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_output, labels=tf.ones_like(fake_output)
        )
    )

# --- Training functions --- #
def train_epoch(epoch, samples, labels, generator_model, discriminator_model,
                generator_optimizer, discriminator_optimizer,
                batch_size, use_time, D_rounds, G_rounds, seq_length, latent_dim, num_signals):
    """
    Train generator and discriminator for one epoch.
    
    Args:
        epoch: Current epoch number
        samples: Training data samples
        labels: Training data labels
        generator_model: Generator model
        discriminator_model: Discriminator model
        generator_optimizer: Optimizer for generator
        discriminator_optimizer: Optimizer for discriminator
        batch_size: Batch size for training
        use_time: Whether to use time information in latent space
        D_rounds: Number of discriminator training rounds per batch
        G_rounds: Number of generator training rounds per batch
        seq_length: Length of time series sequence
        latent_dim: Dimension of latent space
        num_signals: Number of signals to generate
        
    Returns:
        Tuple of (discriminator_loss, generator_loss) as average over the epoch
    """
    total_d_loss = 0
    total_g_loss = 0
    batches = 0
    
    # Calculate number of available batches
    num_batches = len(samples) // batch_size - (D_rounds + G_rounds)
    
    for batch_idx in range(0, num_batches, D_rounds + G_rounds):
        # Get real data batch
        X_mb, Y_mb = data_utils.get_batch(samples, batch_size, batch_idx, labels)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        
        # Train discriminator
        for _ in range(D_rounds):
            Z_mb = sample_Z(batch_size, seq_length, latent_dim, use_time)
            
            with tf.GradientTape() as disc_tape:
                # Generate fake samples
                generated_samples = generator_model(Z_mb, training=True)
                
                # Get discriminator outputs
                real_output = discriminator_model(X_mb, training=True)
                fake_output = discriminator_model(generated_samples, training=True)
                
                # Calculate discriminator loss
                disc_loss = discriminator_loss(real_output, fake_output)
            
            # Apply gradients to discriminator
            disc_gradients = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator_model.trainable_variables))
        
        # Train generator
        for _ in range(G_rounds):
            Z_mb = sample_Z(batch_size, seq_length, latent_dim, use_time)
            
            with tf.GradientTape() as gen_tape:
                # Generate fake samples
                generated_samples = generator_model(Z_mb, training=True)
                
                # Get discriminator output for fake samples
                fake_output = discriminator_model(generated_samples, training=True)
                
                # Calculate generator loss
                gen_loss = generator_loss(fake_output)
            
            # Apply gradients to generator
            gen_gradients = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
            generator_optimizer.apply_gradients(zip(gen_gradients, generator_model.trainable_variables))
        
        # Calculate losses for reporting
        Z_mb = sample_Z(batch_size, seq_length, latent_dim, use_time)
        generated_samples = generator_model(Z_mb, training=False)
        real_output = discriminator_model(X_mb, training=False)
        fake_output = discriminator_model(generated_samples, training=False)
        
        d_loss = discriminator_loss(real_output, fake_output)
        g_loss = generator_loss(fake_output)
        
        total_d_loss += d_loss
        total_g_loss += g_loss
        batches += 1
        
        # Display progress
        #display_batch_progression(batch_idx, num_batches)
    
    # Return average losses
    if batches > 0:
        return float(total_d_loss / batches), float(total_g_loss / batches)
    else:
        return 0.0, 0.0

# --- Utility functions --- #
def display_batch_progression(j, id_max):
    """Display batch progression as percentage."""
    sys.stdout.write(f"{(j / id_max) * 100:.2f}% epoch\r")
    sys.stdout.flush()

# --- Model saving/loading --- #
def save_models(generator_model, discriminator_model, identifier):
    """
    Save generator and discriminator models.
    
    Args:
        generator_model: Generator model to save
        discriminator_model: Discriminator model to save
        identifier: Unique identifier for saved models
        
    Returns:
        Boolean indicating success
    """
    try:
        generator_model.save_weights(f'./experiments/parameters/generator_{identifier}')
        discriminator_model.save_weights(f'./experiments/parameters/discriminator_{identifier}')
        print(f'Models saved: {identifier}')
        return True
    except Exception as e:
        print(f"Error saving models: {e}")
        return False

def load_models(generator_model, discriminator_model, identifier):
    """
    Load generator and discriminator models.
    
    Args:
        generator_model: Generator model to load weights into
        discriminator_model: Discriminator model to load weights into
        identifier: Unique identifier for saved models
        
    Returns:
        Tuple of (generator_model, discriminator_model) with loaded weights
    """
    try:
        generator_model.load_weights(f'./experiments/parameters/generator_{identifier}')
        discriminator_model.load_weights(f'./experiments/parameters/discriminator_{identifier}')
        print(f'Models loaded: {identifier}')
        return generator_model, discriminator_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return generator_model, discriminator_model

# --- Sample generation --- #
def generate_samples(generator_model, batch_size, seq_length, latent_dim, use_time=False):
    """
    Generate samples using the generator model.
    
    Args:
        generator_model: Trained generator model
        batch_size: Number of samples to generate
        seq_length: Length of time series sequence
        latent_dim: Dimension of latent space
        use_time: Whether to use time information in latent space
        
    Returns:
        Generated samples as numpy array
    """
    z = sample_Z(batch_size, seq_length, latent_dim, use_time)
    return generator_model(z, training=False).numpy()