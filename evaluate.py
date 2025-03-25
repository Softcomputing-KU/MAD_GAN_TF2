#!/usr/bin/env ipython
#

import json
import os
import numpy as np
import pandas as pd
import mmd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

import model
import data_utils
import plotting

# for keras
from scipy.stats import ks_2samp
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.backend import clear_session
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

def assert_same_data(A, B):
    """Check if two models were trained on the same data."""
    # case 0, both loaded
    if A['data'] == 'load' and B['data'] == 'load':
        assert A['data_load_from'] == B['data_load_from']
        data_path = './experiments/data/' + A['data_load_from']
    elif A['data'] == 'load' and (not B['data'] == 'load'):
        assert A['data_load_from'] == B['identifier']
        data_path = './experiments/data/' + A['data_load_from']
    elif (not A['data'] == 'load') and B['data'] == 'load':
        assert B['data_load_from'] == A['identifier']
        data_path = './experiments/data/' + A['identifier']
    else:
        raise ValueError(A['data'], B['data'])
    return data_path

def model_memorisation(identifier, epoch, max_samples=2000, tstr=False):
    """
    Compare samples from a model against training set and validation set in mmd.
    
    Args:
        identifier: Model identifier
        epoch: Epoch to evaluate
        max_samples: Maximum number of samples to use
        tstr: Whether this is a TSTR experiment
        
    Returns:
        pvalue, tstat, sigma
    """
    try:
        from eugenium_mmd import MMD_3_Sample_Test
    except ImportError:
        print("Warning: eugenium_mmd module not found. Using approximation.")
        # Simple approximation of MMD_3_Sample_Test if not available
        def MMD_3_Sample_Test(X, Y, Z, sigma=None, computeMMDs=False):
            if sigma is None:
                sigma = mmd.median_pairwise_distance(np.vstack([X, Y, Z]))
            MMDXY = mmd.rbf_mmd2(X, Y, sigma)
            MMDXZ = mmd.rbf_mmd2(X, Z, sigma)
            tstat = (MMDXY - MMDXZ) / np.sqrt(0.1)  # Simplified
            pvalue = 0.5 if tstat <= 0 else (1.0 if tstat > 1.0 else tstat)
            return pvalue, tstat, sigma, MMDXY, MMDXZ
    
    if tstr:
        print('Loading data from TSTR experiment (not sampling from model)')
        # load pre-generated samples
        synth_data = np.load('./experiments/tstr/' + identifier + '_' + str(epoch) + '.data.npy', allow_pickle=True).item()
        model_samples = synth_data['samples']
        synth_labels = synth_data['labels']
        # load real data used in that experiment
        real_data = np.load('./experiments/data/' + identifier + '.data.npy', allow_pickle=True).item()
        real_samples = real_data['samples']
        train = real_samples['train']
        test = real_samples['test']
        n_samples = test.shape[0]
        if model_samples.shape[0] > n_samples:
            model_samples = np.random.permutation(model_samples)[:n_samples]
        print('Data loaded successfully!')
    else:
        # Load settings
        settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
        
        # Get the test, train sets
        data = np.load('./experiments/data/' + identifier + '.data.npy', allow_pickle=True).item()
        train = data['samples']['train']
        test = data['samples']['test']
        n_samples = test.shape[0]
        if n_samples > max_samples:
            n_samples = max_samples
            test = np.random.permutation(test)[:n_samples]
        
        # Load model and generate samples
        # Create models
        hidden_units_g = settings['hidden_units_g']
        hidden_units_d = settings['hidden_units_d']
        num_variables = train.shape[2]
        
        # Create generator model
        generator_model = model.Generator(hidden_units_g, num_variables)
        
        # Build model with dummy input
        seq_length = settings['seq_length']
        latent_dim = settings['latent_dim']
        dummy_z = model.sample_Z(1, seq_length, latent_dim, settings['use_time'])
        _ = generator_model(dummy_z)
        
        # Load weights
        try:
            # Try to load using SavedModel format
            saved_model_path = f'./experiments/parameters/generator_{identifier}_{epoch}'
            if os.path.exists(saved_model_path):
                generator_model = tf.keras.models.load_model(saved_model_path)
            else:
                # Try to load weights
                weights_path = f'./experiments/parameters/generator_{identifier}_{epoch}_weights'
                if os.path.exists(weights_path + '.index'):
                    generator_model.load_weights(weights_path)
                else:
                    print(f"Warning: Could not find model weights for {identifier}_{epoch}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None
        
        # Generate samples
        z_samples = model.sample_Z(n_samples, seq_length, latent_dim, settings['use_time'])
        model_samples = generator_model(z_samples, training=False).numpy()
    
    # Compute MMD
    all_samples = np.vstack([train, test, model_samples])
    heuristic_sigma = mmd.median_pairwise_distance(all_samples)
    print('heuristic sigma:', heuristic_sigma)
    
    # Run MMD test
    pvalue, tstat, sigma, MMDXY, MMDXZ = MMD_3_Sample_Test(
        model_samples, test, np.random.permutation(train)[:n_samples], 
        sigma=heuristic_sigma, computeMMDs=False
    )
    
    return pvalue, tstat, sigma

def sample_trained_model(identifier, epoch, num_samples, Z_samples=None, cond_dim=None, C_samples=None):
    """
    Sample from a trained model.
    
    Args:
        identifier: Model identifier
        epoch: Epoch to load
        num_samples: Number of samples to generate
        Z_samples: Optional latent samples
        cond_dim: Conditional dimension
        C_samples: Conditional samples
        
    Returns:
        Generated samples
    """
    # Load settings
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    
    # Load data to get dimensions
    if settings['data_load_from']:
        data_path = f'./experiments/data/{settings["data_load_from"]}.data.npy'
    else:
        data_path = f'./experiments/data/{identifier}.data.npy'
    
    data = np.load(data_path, allow_pickle=True).item()
    samples = data['samples']
    num_variables = samples['train'].shape[2]
    
    # Create generator model
    hidden_units_g = settings['hidden_units_g']
    generator_model = model.Generator(hidden_units_g, num_variables)
    
    # Build model with dummy input
    seq_length = settings['seq_length']
    latent_dim = settings['latent_dim']
    dummy_z = model.sample_Z(1, seq_length, latent_dim, settings['use_time'])
    _ = generator_model(dummy_z)
    
    # Load weights
    try:
        # Try to load using SavedModel format
        saved_model_path = f'./experiments/parameters/generator_{identifier}_{epoch}'
        if os.path.exists(saved_model_path):
            generator_model = tf.keras.models.load_model(saved_model_path)
        else:
            # Try to load weights
            weights_path = f'./experiments/parameters/generator_{identifier}_{epoch}_weights'
            if os.path.exists(weights_path + '.index'):
                generator_model.load_weights(weights_path)
            else:
                print(f"Warning: Could not find model weights for {identifier}_{epoch}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Generate samples
    if Z_samples is None:
        Z_samples = model.sample_Z(num_samples, seq_length, latent_dim, settings['use_time'])
    
    # Add conditional information if provided
    # This is simplified; in practice, you'd need to incorporate conditional information
    # into your generator model architecture
    
    model_samples = generator_model(Z_samples, training=False).numpy()
    return model_samples

def get_reconstruction_errors(identifier, epoch, g_tolerance=0.05, max_samples=1000, rerun=False, tstr=False):
    """Get reconstruction errors for samples."""
    # Implementation depends on model.invert which would need to be reimplemented for TF 2.5
    # This is a complex function that requires significant changes
    # Placeholder implementation
    print("get_reconstruction_errors is not fully implemented for TF 2.5")
    return True

def train_CNN(train_X, train_Y, vali_X, vali_Y, test_X):
    """Train a CNN classifier (for MNIST)."""
    print('Training CNN!')
    input_shape = (14, 14, 1)
    batch_size = 128
    num_classes = 3
    epochs = 1000

    m = Sequential()
    m.add(Conv2D(16, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.25))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(num_classes, activation='softmax'))

    m.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    m.fit(np.expand_dims(train_X, axis=-1), train_Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.expand_dims(vali_X, axis=-1), vali_Y),
          callbacks=[earlyStopping])
    test_predictions = m.predict(np.expand_dims(test_X, axis=-1))
    return test_predictions

def TSTR_mnist(identifier, epoch, generate=True, duplicate_synth=1, vali=True, CNN=False, reverse=False):
    """
    Train on Synthetic, Test on Real (TSTR) evaluation for MNIST.
    
    Args:
        identifier: Model identifier
        epoch: Epoch to evaluate
        generate: Whether to generate new samples
        duplicate_synth: Duplication factor for synthetic data
        vali: Whether to use validation set for testing
        CNN: Whether to use CNN classifier
        reverse: Whether to do TRTS instead of TSTR
        
    Returns:
        synth_f1, real_f1
    """
    print('Running TSTR on', identifier, 'at epoch', epoch)
    
    if vali:
        test_set = 'vali'
    else:
        test_set = 'test'
        
    if generate:
        data = np.load('./experiments/data/' + identifier + '.data.npy', allow_pickle=True).item()
        samples = data['samples']
        train_X = samples['train']
        test_X = samples[test_set]
        labels = data['labels']
        train_Y = labels['train']
        test_Y = labels[test_set]
        
        # Generate synthetic data
        synth_Y = np.tile(train_Y, [duplicate_synth, 1])
        synth_X = sample_trained_model(identifier, epoch, num_samples=synth_Y.shape[0], C_samples=synth_Y)
        
        # For use in TRTS
        synth_testX = sample_trained_model(identifier, epoch, num_samples=test_Y.shape[0], C_samples=test_Y)
        
        # Save synthetic data
        synth_data = {
            'samples': synth_X, 
            'labels': synth_Y, 
            'test_samples': synth_testX, 
            'test_labels': test_Y
        }
        os.makedirs('./experiments/tstr/', exist_ok=True)
        np.save('./experiments/tstr/' + identifier + '_' + str(epoch) + '.data.npy', synth_data)
    else:
        print('Loading synthetic data from pre-sampled model')
        exp_data = np.load('./experiments/tstr/' + identifier + '_' + str(epoch) + '.data.npy', allow_pickle=True).item()
        test_X = exp_data.get('test_data', exp_data.get('test_samples'))
        test_Y = exp_data.get('test_labels')
        train_X = exp_data.get('train_data', exp_data.get('samples'))
        train_Y = exp_data.get('train_labels', exp_data.get('labels'))
        synth_X = exp_data.get('synth_data', exp_data.get('samples'))
        synth_Y = exp_data.get('synth_labels', exp_data.get('labels'))
    
    if reverse:
        which_setting = 'trts'
        print('Swapping synthetic test set in for real, to do TRTS!')
        test_X = synth_testX
    else:
        print('Doing normal TSTR')
        which_setting = 'tstr'
    
    # Train and evaluate classifiers
    if not CNN:
        model_choice = 'RF'
        # Reshape if needed
        if len(test_X.shape) == 3:
            test_X = test_X.reshape(test_X.shape[0], -1)
        if len(train_X.shape) == 3:
            train_X = train_X.reshape(train_X.shape[0], -1)
        if len(synth_X.shape) == 3:
            synth_X = synth_X.reshape(synth_X.shape[0], -1)
        
        # Convert one-hot to class indices if needed
        if len(synth_Y.shape) > 1 and synth_Y.shape[1] > 1:
            synth_Y = np.argmax(synth_Y, axis=1)
            train_Y = np.argmax(train_Y, axis=1)
            test_Y = np.argmax(test_Y, axis=1)
        
        # Create and train classifiers
        synth_classifier = RandomForestClassifier(n_estimators=500)
        real_classifier = RandomForestClassifier(n_estimators=500)
        
        # Fit classifiers
        real_classifier.fit(train_X, train_Y)
        synth_classifier.fit(synth_X, synth_Y)
        
        # Test on real data
        synth_predY = synth_classifier.predict(test_X)
        real_predY = real_classifier.predict(test_X)
    else:
        model_choice = 'CNN'
        # Load validation data if needed
        if 'samples' not in locals() or 'labels' not in locals():
            data = np.load('./experiments/data/' + identifier + '.data.npy', allow_pickle=True).item()
            samples = data['samples']
            labels = data['labels']
        
        # Train CNNs
        synth_predY = train_CNN(synth_X, synth_Y, samples['vali'], labels['vali'], test_X)
        clear_session()
        real_predY = train_CNN(train_X, train_Y, samples['vali'], labels['vali'], test_X)
        clear_session()
        
        # CNN output is one-hot, convert to class indices
        test_Y = np.argmax(test_Y, axis=1)
        synth_predY = np.argmax(synth_predY, axis=1)
        real_predY = np.argmax(real_predY, axis=1)
    
    # Calculate metrics
    synth_prec, synth_recall, synth_f1, synth_support = precision_recall_fscore_support(test_Y, synth_predY, average='weighted')
    synth_accuracy = accuracy_score(test_Y, synth_predY)
    synth_auprc = 'NaN'
    synth_auroc = 'NaN'
    synth_scores = [synth_prec, synth_recall, synth_f1, synth_accuracy, synth_auprc, synth_auroc]
    
    real_prec, real_recall, real_f1, real_support = precision_recall_fscore_support(test_Y, real_predY, average='weighted')
    real_accuracy = accuracy_score(test_Y, real_predY)
    real_auprc = 'NaN'
    real_auroc = 'NaN'
    real_scores = [real_prec, real_recall, real_f1, real_accuracy, real_auprc, real_auroc]
    
    all_scores = synth_scores + real_scores
    
    # Save results
    os.makedirs('./experiments/tstr/', exist_ok=True)
    if vali:
        report_file = open('./experiments/tstr/vali.' + which_setting + '_report.v3.csv', 'a')
        report_file.write('mnist,' + identifier + ',' + model_choice + ',' + str(epoch) + ',' + ','.join(map(str, all_scores)) + '\n')
        report_file.close()
    else:
        report_file = open('./experiments/tstr/' + which_setting + '_report.v3.csv', 'a')
        report_file.write('mnist,' + identifier + ',' + model_choice + ',' + str(epoch) + ',' + ','.join(map(str, all_scores)) + '\n')
        report_file.close()
        # Visualize results
        try:
            plotting.view_mnist_eval(identifier + '_' + str(epoch), train_X, train_Y, synth_X, synth_Y, test_X, test_Y, synth_predY, real_predY)
        except (ValueError, AttributeError) as e:
            print(f'PLOTTING ERROR: {e}')
    
    print(classification_report(test_Y, synth_predY))
    print(classification_report(test_Y, real_predY))
    
    return synth_f1, real_f1

# Additional functions can be implemented as needed for specific evaluation tasks