import numpy as np
import tensorflow as tf
import random
import json
import data_utils
import plotting
import model
import utils

from scipy.stats import mode
from time import time
from math import floor
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio

begin = time()
tf.get_logger().setLevel('ERROR')  # Suppress warning messages

# --- Get settings --- #
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
if settings['settings_file']: 
    settings = utils.load_settings_from_file(settings)

# --- Load data --- #
data_path = f'./experiments/data/{settings["data_load_from"]}.data.npy'
print(f'Loading data from {data_path}')
settings["eval_an"] = False
settings["eval_single"] = False

samples, labels, index = data_utils.get_data(
    settings["data"], settings["seq_length"], settings["seq_step"],
    settings["num_signals"], settings["sub_id"], settings["eval_single"],
    settings["eval_an"], data_path
)
print(f'samples_size: {samples.shape}')
num_variables = samples.shape[2]
print(f'num_variables: {num_variables}')

# --- Save settings --- #
print('Ready to run with settings:')
for k, v in settings.items():
    print(f'{v}\t{k}')
locals().update(settings)
json.dump(settings, open(f'./experiments/settings/{identifier}.txt', 'w'), indent=0)

# --- Create models --- #
# Create generator and discriminator models
generator_model = model.Generator(settings['hidden_units_g'], num_variables)
discriminator_model = model.Discriminator(settings['hidden_units_d'])

# Create optimizers
generator_optimizer = tf.keras.optimizers.Adam(settings['learning_rate'])
discriminator_optimizer = tf.keras.optimizers.Adam(settings['learning_rate'])

# Build models with a sample input
dummy_z = model.sample_Z(1, settings['seq_length'], settings['latent_dim'], settings['use_time'])
dummy_x = tf.zeros([1, settings['seq_length'], num_variables], dtype=tf.float32)
_ = generator_model(dummy_z)
_ = discriminator_model(dummy_x)
print("Models created successfully")

# --- Visualize real samples --- #
vis_real_indices = np.random.choice(len(samples), size=16)
vis_real = np.float32(samples[vis_real_indices, :, :])
plotting.save_plot_sample(vis_real, 0, f'{identifier}_real', n_samples=16, num_epochs=settings['num_epochs'])
plotting.save_samples_real(vis_real, identifier)

# --- Training loop --- #
MMD = np.zeros([settings['num_epochs'], ])
t0 = time()

for epoch in range(settings['num_epochs']):
    # Train for one epoch
    D_loss_curr, G_loss_curr = model.train_epoch(
        epoch, samples, labels, generator_model, discriminator_model,
        generator_optimizer, discriminator_optimizer, settings['batch_size'], 
        settings['use_time'], settings['D_rounds'], settings['G_rounds'], 
        settings['seq_length'], settings['latent_dim'], num_variables
    )
    
    print(f'Epoch {epoch}, D_loss: {D_loss_curr:.4f}, G_loss: {G_loss_curr:.4f}, seq_length: {settings["seq_length"]}')
    
    # Visualize samples at intervals
    if (epoch + 1) % 10 == 0 or epoch == 0:
        z_vis = model.sample_Z(16, settings['seq_length'], settings['latent_dim'], settings['use_time'])
        generated_samples = generator_model(z_vis, training=False).numpy()
        plotting.save_plot_sample(generated_samples, epoch+1, identifier, n_samples=16, num_epochs=settings['num_epochs'])
    
    # Save models at intervals
    if (epoch + 1) % 10 == 0 or epoch == settings['num_epochs'] - 1:
        model.save_models(generator_model, discriminator_model, f'{settings["sub_id"]}_{settings["seq_length"]}_{epoch}')

# --- Save results --- #
np.save(f'./experiments/plots/gs/{identifier}_MMD.npy', MMD)

# --- Generate final samples --- #
z_final = model.sample_Z(16, settings['seq_length'], settings['latent_dim'], settings['use_time'])
final_samples = generator_model(z_final, training=False).numpy()
plotting.save_samples_generated(final_samples, identifier)

# --- Summary --- #
end = time() - begin
print(f'Training completed | Total time: {int(end)} seconds')
print(f'Final D_loss: {D_loss_curr:.4f}, G_loss: {G_loss_curr:.4f}')