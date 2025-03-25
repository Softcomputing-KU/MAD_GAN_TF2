import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import model
import mmd
from sklearn.metrics import precision_recall_fscore_support
import os
import json
import traceback

def anomaly_detection_plot(D_test, T_mb, L_mb, D_L, epoch, identifier):
    """
    Plot anomaly detection results for visualization
    
    Args:
        D_test: Discriminator test outputs
        T_mb: Test samples
        L_mb: Labels
        D_L: Detection decisions
        epoch: Current epoch
        identifier: Model identifier
    """
    try:
        aa = D_test.shape[0]
        bb = D_test.shape[1]
        D_L = D_L.reshape([aa, bb, -1])

        x_points = np.arange(bb)

        fig, ax = plt.subplots(4, 4, sharex=True)
        for m in range(4):
            for n in range(4):
                D = D_test[n * 4 + m, :, :]
                T = T_mb[n * 4 + m, :, :]
                L = L_mb[n * 4 + m, :, :]
                DL = D_L[n * 4 + m, :, :]
                ax[m, n].plot(x_points, D, '--g', label='Pro')
                ax[m, n].plot(x_points, T, 'b', label='Data')
                ax[m, n].plot(x_points, L, 'k', label='Label')
                ax[m, n].plot(x_points, DL, 'r', label='Label')
                ax[m, n].set_ylim(-1, 1)
        for n in range(4):
            ax[-1, n].xaxis.set_ticks(range(0, bb, int(bb/6)))
        fig.suptitle(f"Epoch {epoch}")
        fig.subplots_adjust(hspace=0.15)
        
        # Ensure directory exists
        os.makedirs("./experiments/plots/DR_dis/", exist_ok=True)
        
        fig.savefig(f"./experiments/plots/DR_dis/{identifier}_epoch{str(epoch).zfill(4)}.png")
        plt.clf()
        plt.close()
        return True
    except Exception as e:
        print(f"Error in anomaly_detection_plot: {e}")
        traceback.print_exc()
        return False

def detection_D_I(DD, L_mb, I_mb, seq_step, tao):
    """
    Detect anomalies using discriminator outputs.
    
    Args:
        DD: Discriminator outputs
        L_mb: Labels
        I_mb: Indices
        seq_step: Sequence step size
        tao: Detection threshold
        
    Returns:
        Accuracy, precision, recall, F1 score, false positive rate, and detection decisions
    """
    try:
        # Ensure DD is numpy array
        DD = np.array(DD)
        L_mb = np.array(L_mb)
        I_mb = np.array(I_mb)
        
        print(f"DD shape: {DD.shape}, L_mb shape: {L_mb.shape}, I_mb shape: {I_mb.shape}")
        
        # Get dimensions
        aa = DD.shape[0]
        bb = DD.shape[1]

        # Calculate total length
        LL = (aa-1)*seq_step+bb

        # Reshape inputs
        DD = abs(DD.reshape([aa, bb]))
        L_mb = L_mb.reshape([aa, bb])
        I_mb = I_mb.reshape([aa, bb])
        
        # Initialize arrays for detection
        D_L = np.zeros([LL, 1])
        L_L = np.zeros([LL, 1])
        Count = np.zeros([LL, 1])
        
        # Aggregate values by time step
        for i in range(0, aa):
            for j in range(0, bb):
                # Changed from i*10+j to i*seq_step+j for consistency
                D_L[i*seq_step+j] += DD[i, j]
                L_L[i*seq_step+j] += L_mb[i, j]
                Count[i*seq_step+j] += 1

        # Calculate average values
        D_L /= Count
        L_L /= Count

        # Initialize counters for confusion matrix
        TP, TN, FP, FN = 0, 0, 0, 0

        # Apply threshold and calculate confusion matrix
        for i in range(LL):
            if D_L[i] > tao:
                # true/negative
                D_L[i] = 0
            else:
                # false/positive
                D_L[i] = 1

            A = D_L[i]
            B = L_L[i]
            if A == 1 and B == 1:
                TP += 1
            elif A == 1 and B == 0:
                FP += 1
            elif A == 0 and B == 0:
                TN += 1
            elif A == 0 and B == 1:
                FN += 1

        # Calculate accuracy
        cc = (D_L == L_L)
        cc = list(cc.reshape([-1]))
        N = cc.count(True)
        Accu = float((N / LL) * 100)

        # Calculate metrics using sklearn
        precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')
        
        # Calculate false positive rate
        FPR = (100 * FP) / (FP + TN) if (FP + TN) > 0 else 0

        return Accu, precision, recall, f1, FPR, D_L
    
    except Exception as e:
        print(f"Error in detection_D_I: {e}")
        traceback.print_exc()
        # Return default values in case of error
        return 0, 0, 0, 0, 0, np.zeros((aa*seq_step, 1))

def detection_R_D_I(DD, Gs, T_mb, L_mb, seq_step, tao, lam):
    """
    Detect anomalies using both reconstruction error and discriminator outputs.
    
    Args:
        DD: Discriminator outputs
        Gs: Generated (reconstructed) samples
        T_mb: Test samples
        L_mb: Labels
        seq_step: Sequence step size
        tao: Detection threshold
        lam: Weight parameter for combining discriminator and reconstruction scores
        
    Returns:
        Accuracy, precision, recall, F1 score, false positive rate, and detection decisions
    """
    try:
        # Ensure inputs are numpy arrays
        DD = np.array(DD)
        Gs = np.array(Gs)
        T_mb = np.array(T_mb)
        L_mb = np.array(L_mb)
        
        print(f"DD shape: {DD.shape}, Gs shape: {Gs.shape}, T_mb shape: {T_mb.shape}, L_mb shape: {L_mb.shape}")
        
        # Handle different dimensions
        if len(Gs.shape) == 2 and len(T_mb.shape) == 3:
            # Gs is 2D but T_mb is 3D - reshape one to match the other
            if T_mb.shape[1] == 1:
                # If T_mb has seq_length=1, reshape to 2D
                T_mb = T_mb.reshape(T_mb.shape[0], T_mb.shape[2])
            else:
                # If T_mb has seq_length>1, reshape Gs to match
                Gs = Gs.reshape(Gs.shape[0], 1, Gs.shape[1])
        elif len(Gs.shape) == 3 and len(T_mb.shape) == 2:
            # T_mb is 2D but Gs is 3D
            if Gs.shape[1] == 1:
                # If Gs has seq_length=1, reshape to 2D
                Gs = Gs.reshape(Gs.shape[0], Gs.shape[2])
            else:
                # If Gs has seq_length>1, reshape T_mb to match
                T_mb = T_mb.reshape(T_mb.shape[0], 1, T_mb.shape[1])
        
        print(f"After reshaping: Gs shape: {Gs.shape}, T_mb shape: {T_mb.shape}")
        
        # Calculate reconstruction error
        if len(Gs.shape) == 3 and len(T_mb.shape) == 3:
            R = np.absolute(Gs - T_mb)
            R = np.mean(R, axis=2)
        else:
            # Both are 2D
            R = np.absolute(Gs - T_mb)
            R = np.mean(R, axis=1).reshape(-1, 1)
        
        # Get dimensions
        aa = DD.shape[0]
        bb = DD.shape[1]

        # Calculate total length
        LL = (aa - 1) * seq_step + bb

        # Reshape inputs
        DD = abs(DD.reshape([aa, bb]))
        DD = 1 - DD  # Invert probability (higher value = more likely anomaly)
        L_mb = L_mb.reshape([aa, bb])
        R = R.reshape([aa, bb]) if len(R.shape) > 1 and R.shape[0] == aa and R.shape[1] == bb else np.tile(R.reshape([aa, 1]), [1, bb])

        # Initialize arrays for detection
        D_L = np.zeros([LL, 1])
        R_L = np.zeros([LL, 1])
        L_L = np.zeros([LL, 1])
        L_pre = np.zeros([LL, 1])
        Count = np.zeros([LL, 1])
        
        # Aggregate values by time step
        for i in range(0, aa):
            for j in range(0, bb):
                # Changed from i*10+j to i*seq_step+j for consistency
                D_L[i*seq_step+j] += DD[i, j]
                L_L[i*seq_step+j] += L_mb[i, j]
                R_L[i*seq_step+j] += R[i, j]
                Count[i*seq_step+j] += 1
        
        # Calculate average values
        D_L /= Count
        L_L /= Count
        R_L /= Count

        # Normalize R_L and D_L for better combination
        if np.max(R_L) - np.min(R_L) > 0:
            R_L = (R_L - np.min(R_L)) / (np.max(R_L) - np.min(R_L))
        if np.max(D_L) - np.min(D_L) > 0:
            D_L = (D_L - np.min(D_L)) / (np.max(D_L) - np.min(D_L))

        # Initialize counters for confusion matrix
        TP, TN, FP, FN = 0, 0, 0, 0

        # Apply threshold and calculate confusion matrix
        for i in range(LL):
            # Combine scores with lambda weight
            combined_score = (1-lam)*R_L[i] + lam*D_L[i]
            
            if combined_score > tao:
                # Mark as anomaly
                L_pre[i] = 1
            else:
                # Mark as normal
                L_pre[i] = 0

            A = L_pre[i]
            B = L_L[i]
            if A == 1 and B == 1:
                TP += 1
            elif A == 1 and B == 0:
                FP += 1
            elif A == 0 and B == 0:
                TN += 1
            elif A == 0 and B == 1:
                FN += 1

        # Calculate accuracy
        cc = (L_pre == L_L)
        cc = list(cc.reshape([-1]))
        N = cc.count(True)
        Accu = float((N / LL) * 100)

        # Calculate metrics
        # Avoid division by zero
        Pre = (100 * TP) / (TP + FP) if (TP + FP) > 0 else 0
        Rec = (100 * TP) / (TP + FN) if (TP + FN) > 0 else 0
        F1 = (2 * Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        FPR = (100 * FP) / (FP + TN) if (FP + TN) > 0 else 0

        return Accu, Pre/100, Rec/100, F1/100, FPR, L_pre
    
    except Exception as e:
        print(f"Error in detection_R_D_I: {e}")
        traceback.print_exc()
        # Return default values in case of error
        return 0, 0, 0, 0, 0, np.zeros((LL, 1))

def detection_R_I(Gs, T_mb, L_mb, seq_step, tao):
    """
    Detect anomalies using only reconstruction error.
    
    Args:
        Gs: Generated (reconstructed) samples
        T_mb: Test samples
        L_mb: Labels
        seq_step: Sequence step size
        tao: Detection threshold
        
    Returns:
        Accuracy, precision, recall, F1 score, false positive rate, and detection decisions
    """
    try:
        # Ensure inputs are numpy arrays
        Gs = np.array(Gs)
        T_mb = np.array(T_mb)
        L_mb = np.array(L_mb)
        
        print(f"Gs shape: {Gs.shape}, T_mb shape: {T_mb.shape}, L_mb shape: {L_mb.shape}")
        
        # Handle different dimensions
        if len(Gs.shape) == 2 and len(T_mb.shape) == 3:
            # Gs is 2D but T_mb is 3D - reshape one to match the other
            if T_mb.shape[1] == 1:
                # If T_mb has seq_length=1, reshape to 2D
                T_mb = T_mb.reshape(T_mb.shape[0], T_mb.shape[2])
            else:
                # If T_mb has seq_length>1, reshape Gs to match
                Gs = Gs.reshape(Gs.shape[0], 1, Gs.shape[1])
        elif len(Gs.shape) == 3 and len(T_mb.shape) == 2:
            # T_mb is 2D but Gs is 3D
            if Gs.shape[1] == 1:
                # If Gs has seq_length=1, reshape to 2D
                Gs = Gs.reshape(Gs.shape[0], Gs.shape[2])
            else:
                # If Gs has seq_length>1, reshape T_mb to match
                T_mb = T_mb.reshape(T_mb.shape[0], 1, T_mb.shape[1])
                
        print(f"After reshaping: Gs shape: {Gs.shape}, T_mb shape: {T_mb.shape}")
        
        # Calculate reconstruction error
        if len(Gs.shape) == 3 and len(T_mb.shape) == 3:
            R = np.absolute(Gs - T_mb)
            R = np.mean(R, axis=2)
        else:
            # Both are 2D
            R = np.absolute(Gs - T_mb)
            R = np.mean(R, axis=1).reshape(-1, 1)
        
        # Get dimensions
        aa = Gs.shape[0]
        bb = 1 if len(Gs.shape) == 2 else Gs.shape[1]

        # Calculate total length
        LL = (aa - 1) * seq_step + bb

        # Reshape inputs
        L_mb = L_mb.reshape([aa, -1])  # Allow flexible second dimension
        if len(R.shape) == 2 and R.shape[1] == bb:
            # R is already the right shape
            pass
        else:
            # Reshape R to match expected dimensions
            R = np.tile(R.reshape([aa, 1]), [1, bb])

        # Initialize arrays for detection
        L_L = np.zeros([LL, 1])
        R_L = np.zeros([LL, 1])
        L_pre = np.zeros([LL, 1])
        Count = np.zeros([LL, 1])
        
        # Aggregate values by time step
        for i in range(0, aa):
            for j in range(0, min(bb, L_mb.shape[1])):
                # Changed from i*10+j to i*seq_step+j for consistency
                L_L[i*seq_step+j] += L_mb[i, j]
                R_L[i*seq_step+j] += R[i, j] if j < R.shape[1] else 0
                Count[i*seq_step+j] += 1
        
        # Calculate average values
        L_L /= Count
        R_L /= Count
        
        # Normalize R_L for better thresholding
        if np.max(R_L) - np.min(R_L) > 0:
            R_L = (R_L - np.min(R_L)) / (np.max(R_L) - np.min(R_L))

        # Initialize counters for confusion matrix
        TP, TN, FP, FN = 0, 0, 0, 0

        # Apply threshold and calculate confusion matrix
        for i in range(LL):
            if R_L[i] > tao:
                # Mark as anomaly
                L_pre[i] = 1
            else:
                # Mark as normal
                L_pre[i] = 0

            A = L_pre[i]
            B = L_L[i]
            if A == 1 and B == 1:
                TP += 1
            elif A == 1 and B == 0:
                FP += 1
            elif A == 0 and B == 0:
                TN += 1
            elif A == 0 and B == 1:
                FN += 1

        # Calculate accuracy
        cc = (L_pre == L_L)
        cc = list(cc.reshape([-1]))
        N = cc.count(True)
        Accu = float((N / LL) * 100)

        # Calculate metrics
        # Avoid division by zero
        Pre = (100 * TP) / (TP + FP) if (TP + FP) > 0 else 0
        Rec = (100 * TP) / (TP + FN) if (TP + FN) > 0 else 0
        F1 = (2 * Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        FPR = (100 * FP) / (FP + TN) if (FP + TN) > 0 else 0

        return Accu, Pre/100, Rec/100, F1/100, FPR, L_pre
    
    except Exception as e:
        print(f"Error in detection_R_I: {e}")
        traceback.print_exc()
        # Return default values in case of error
        return 0, 0, 0, 0, 0, np.zeros((LL, 1))

def dis_D_model(settings, T_mb, para_path):
    """
    Apply discriminator model to test samples.
    
    Args:
        settings: Model settings
        T_mb: Test batch
        para_path: Path to model weights
        
    Returns:
        D_t, L_t: Discriminator outputs
    """
    try:
        print(f"Processing data in dis_D_model, original shape: {np.array(T_mb).shape}")
        
        # Ensure T_mb is numpy array
        if isinstance(T_mb, list):
            T_mb = np.array(T_mb)
        
        # Track original shape for output consistency
        original_shape = T_mb.shape
        
        # Reshape input data if needed
        if len(T_mb.shape) == 1:
            # 1D shape (features) -> 3D (batch=1, seq_len=1, features)
            T_mb = T_mb.reshape(1, 1, -1)
            print(f"Reshaped 1D data to {T_mb.shape}")
        elif len(T_mb.shape) == 2:
            # 2D shape could be (batch, features) or (seq_len, features)
            # Assuming (batch, features)
            T_mb = T_mb.reshape(T_mb.shape[0], 1, T_mb.shape[1])
            print(f"Reshaped 2D data to {T_mb.shape}")
        
        # Check for model file existence
        model_path_parts = para_path.split('/')[-1].split('.')[0].split('_')
        identifier = model_path_parts[0] if len(model_path_parts) > 0 else "unknown"
        seq_length_str = model_path_parts[1] if len(model_path_parts) > 1 else "0"
        epoch = model_path_parts[2] if len(model_path_parts) > 2 else "0"
        model_identifier = f"{identifier}_{seq_length_str}_{epoch}"
        
        print(f"Looking for model weights: {model_identifier}")
        
        # Check various possible file paths
        weights_paths = [
            f'./experiments/parameters/generator_{model_identifier}',
            f'./experiments/parameters/generator_{model_identifier}.h5',
            f'./experiments/parameters/generator_{model_identifier}_weights',
            f'./experiments/parameters/{model_identifier}.npy',
            para_path
        ]
        
        existing_path = None
        for path in weights_paths:
            if os.path.exists(path) or (path.endswith('_weights') and os.path.exists(path + '.index')):
                existing_path = path
                print(f"Found model weights at: {existing_path}")
                break
        
        if existing_path is None:
            print("WARNING: Could not find model weights file. Using untrained model.")
        
        # Determine number of features 
        num_signals = T_mb.shape[-1]
        
        # Create models
        generator_model = model.Generator(settings['hidden_units_g'], num_signals)
        discriminator_model = model.Discriminator(settings['hidden_units_d'])
        
        # Build models with dummy inputs
        seq_length = settings['seq_length']
        latent_dim = settings['latent_dim']
        batch_size = T_mb.shape[0]
        
        dummy_z = model.sample_Z(1, seq_length, latent_dim, settings['use_time'])
        dummy_x = tf.zeros([1, seq_length, num_signals], dtype=tf.float32)
        _ = generator_model(dummy_z)
        _ = discriminator_model(dummy_x)
        
        # Try to load model weights if available
        if existing_path is not None:
            try:
                if existing_path.endswith('.npy'):
                    # Special case for old format
                    parameters = np.load(existing_path, allow_pickle=True).item()
                    print(f"Successfully loaded parameters from {existing_path}")
                    # We can't directly load old format weights in TF 2.x
                    print("WARNING: Using simplified model without loaded weights.")
                else:
                    # Standard TF 2.x weight loading
                    model.load_models(generator_model, discriminator_model, model_identifier)
                    print(f"Successfully loaded model weights from {model_identifier}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Using untrained model.")
        else:
            print("Using untrained model.")
        
        # Process test samples through discriminator
        T_mb_tensor = tf.convert_to_tensor(T_mb, dtype=tf.float32)
        
        # Process in batches if too large to avoid memory issues
        max_batch_size = 128
        if batch_size > max_batch_size:
            all_logits = []
            all_probs = []
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                batch = T_mb_tensor[i:end_idx]
                logits = discriminator_model(batch, training=False)
                probs = tf.nn.sigmoid(logits)
                all_logits.append(logits.numpy())
                all_probs.append(probs.numpy())
            
            D_logits = np.concatenate(all_logits, axis=0)
            D_prob = np.concatenate(all_probs, axis=0)
        else:
            D_logits = discriminator_model(T_mb_tensor, training=False)
            D_prob = tf.nn.sigmoid(D_logits)
            D_logits = D_logits.numpy()
            D_prob = D_prob.numpy()
        
        print(f"Generated discriminator outputs with shape: {D_prob.shape}")
        
        # Format output to match expected shape based on original input
        if len(original_shape) == 1:
            # If input was 1D, return consistent output
            D_prob = D_prob.reshape(-1)
            D_logits = D_logits.reshape(-1)
        
        return D_prob, D_logits
        
    except Exception as e:
        print(f"Error in dis_D_model function: {e}")
        traceback.print_exc()
        # Return dummy outputs with appropriate shape in case of error
        if isinstance(T_mb, (list, np.ndarray)):
            if isinstance(T_mb, list):
                T_mb = np.array(T_mb)
            if len(T_mb.shape) == 1:
                return np.ones(1) * 0.5, np.zeros(1)
            elif len(T_mb.shape) == 2:
                return np.ones((T_mb.shape[0], 1)) * 0.5, np.zeros((T_mb.shape[0], 1))
            else:
                return np.ones((T_mb.shape[0], T_mb.shape[1], 1)) * 0.5, np.zeros((T_mb.shape[0], T_mb.shape[1], 1))
        else:
            return np.array([0.5]), np.array([0])

def invert(settings, T_mb, para_path, g_tolerance=None, e_tolerance=0.1, 
           n_iter=None, max_iter=1000, heuristic_sigma=None):
    """
    Invert generator to find latent representation and reconstruction.
    
    Args:
        settings: Model settings
        T_mb: Test batch
        para_path: Path to model weights
        g_tolerance: Gradient tolerance
        e_tolerance: Error tolerance
        n_iter: Number of iterations
        max_iter: Maximum iterations
        heuristic_sigma: Sigma for MMD
        
    Returns:
        Gs, Zs, error_per_sample, heuristic_sigma
    """
    try:
        # Convert to numpy array if it's a list
        if isinstance(T_mb, list):
            T_mb = np.array(T_mb)
            
        print(f"Inverting data with original shape: {T_mb.shape}")
        
        # Track original shape for output consistency
        original_shape = T_mb.shape
        
        # Ensure T_mb is a float32 numpy array
        T_mb = np.float32(T_mb)
        
        # Reshape input data if needed
        if len(T_mb.shape) == 1:
            # 1D shape (features) -> 3D (batch=1, seq_len=1, features)
            T_mb = T_mb.reshape(1, 1, -1)
            print(f"Reshaped 1D data to {T_mb.shape}")
        elif len(T_mb.shape) == 2:
            # 2D shape could be (batch, features) or (seq_len, features)
            # Assuming (batch, features)
            T_mb = T_mb.reshape(T_mb.shape[0], 1, T_mb.shape[1])
            print(f"Reshaped 2D data to {T_mb.shape}")
        
        # If settings is a string, load settings from file
        if isinstance(settings, str):
            try:
                with open(f'./experiments/settings/{settings}.txt', 'r') as f:
                    settings = json.load(f)
            except Exception as e:
                print(f"Error loading settings from file: {e}")
                # Continue with existing settings
        
        # Check for model file existence
        model_path_parts = para_path.split('/')[-1].split('.')[0].split('_')
        identifier = model_path_parts[0] if len(model_path_parts) > 0 else "unknown"
        seq_length_str = model_path_parts[1] if len(model_path_parts) > 1 else "0"
        epoch = model_path_parts[2] if len(model_path_parts) > 2 else "0"
        model_identifier = f"{identifier}_{seq_length_str}_{epoch}"
        
        print(f"Looking for model weights: {model_identifier}")
        
        # Check various possible file paths
        weights_paths = [
            f'./experiments/parameters/generator_{model_identifier}',
            f'./experiments/parameters/generator_{model_identifier}.h5',
            f'./experiments/parameters/generator_{model_identifier}_weights',
            f'./experiments/parameters/{model_identifier}.npy',
            para_path
        ]
        
        existing_path = None
        for path in weights_paths:
            if os.path.exists(path) or (path.endswith('_weights') and os.path.exists(path + '.index')):
                existing_path = path
                print(f"Found model weights at: {existing_path}")
                break
        
        if existing_path is None:
            print("WARNING: Could not find model weights file. Using untrained model.")
        
        # Determine dimensions 
        num_signals = T_mb.shape[-1]
        batch_size = T_mb.shape[0]
        seq_length = settings.get('seq_length', T_mb.shape[1])
        latent_dim = settings.get('latent_dim', 100)  # Default if not found
        
        # Create models
        generator_model = model.Generator(settings['hidden_units_g'], num_signals)
        discriminator_model = model.Discriminator(settings['hidden_units_d'])
        
        # Build models with dummy inputs
        dummy_z = model.sample_Z(1, seq_length, latent_dim, settings.get('use_time', False))
        dummy_x = tf.zeros([1, seq_length, num_signals], dtype=tf.float32)
        _ = generator_model(dummy_z)
        _ = discriminator_model(dummy_x)
        
        # Try to load model weights if available
        if existing_path is not None:
            try:
                if existing_path.endswith('.npy'):
                    # Special case for old format
                    parameters = np.load(existing_path, allow_pickle=True).item()
                    print(f"Successfully loaded parameters from {existing_path}")
                    # Placeholder for weight initialization
                    print("WARNING: Using simplified model without loaded weights.")
                else:
                    # Standard TF 2.x weight loading
                    model.load_models(generator_model, discriminator_model, model_identifier)
                    print(f"Successfully loaded model weights from {model_identifier}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Using untrained model.")
        else:
            print("Using untrained model.")
        
        # Convert T_mb to tensor
        T_mb_tensor = tf.convert_to_tensor(T_mb, dtype=tf.float32)
        
        # Initialize latent vector
        Z = tf.Variable(
            model.sample_Z(batch_size, seq_length, latent_dim, settings.get('use_time', False)),
            trainable=True
        )
        
        # Compute heuristic sigma if not provided
        if heuristic_sigma is None:
            try:
                heuristic_sigma = mmd.median_pairwise_distance(T_mb)
                print(f"Using heuristic sigma: {heuristic_sigma}")
            except Exception as e:
                print(f"Error computing heuristic sigma: {e}")
                heuristic_sigma = 1.0
        
        # Set optimization parameters
        if n_iter is None:
            n_iter = max_iter
        
        # Use Adam optimizer for inversion
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        
        # Keep track of best results
        best_loss = float('inf')
        best_z = None
        best_g = None
        
        # Optimization loop
        for i in range(n_iter):
            with tf.GradientTape() as tape:
                # Generate samples
                G_samples = generator_model(Z, training=False)
                
                # Compute MSE loss (simpler than MMD for stability)
                mse_loss = tf.reduce_mean(tf.square(T_mb_tensor - G_samples))
            
            # Get gradients and apply updates
            gradients = tape.gradient(mse_loss, [Z])
            optimizer.apply_gradients(zip(gradients, [Z]))
            
            # Track best result
            loss_value = float(mse_loss.numpy())
            if loss_value < best_loss:
                best_loss = loss_value
                best_z = Z.numpy().copy()
                best_g = generator_model(Z, training=False).numpy()
            
            # Check for convergence
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss_value}")
            
            # Early stopping if tolerance is met
            if e_tolerance is not None and loss_value < e_tolerance:
                print(f"Converged at iteration {i} with loss {loss_value}")
                break
        
        # Use best found solution
        if best_z is not None:
            Z_final = best_z
            G_final = best_g
        else:
            # If no better solution was found, use the last one
            G_final = generator_model(Z, training=False).numpy()
            Z_final = Z.numpy()
        
        # Compute error per sample
        error_per_sample = np.zeros(batch_size)
        for i in range(batch_size):
            # Use a simplified error calculation
            error_per_sample[i] = np.mean(np.square(T_mb[i] - G_final[i]))
        
        print(f"Generated output shape before reshaping: {G_final.shape}")
        
        # Reshape output to match expected format based on original input
        if len(original_shape) == 1:
            # Original input was 1D vector
            G_final = G_final.reshape(-1)
        elif len(original_shape) == 2 and len(G_final.shape) == 3 and G_final.shape[1] == 1:
            # Original input was 2D, remove the middle dimension if it's 1
            G_final = G_final.reshape(G_final.shape[0], G_final.shape[2])
        
        print(f"Final output shape: {G_final.shape}")
        return G_final, Z_final, error_per_sample, heuristic_sigma
    
    except Exception as e:
        print(f"Error in invert function: {e}")
        traceback.print_exc()
        # Return dummy outputs with appropriate shape
        if isinstance(T_mb, (list, np.ndarray)):
            if isinstance(T_mb, list):
                T_mb = np.array(T_mb)
            if len(T_mb.shape) == 1:
                dummy_Z = np.zeros((1, 1, settings.get('latent_dim', 100)))
                dummy_G = np.zeros_like(T_mb)
                dummy_errors = np.ones(1)
            elif len(T_mb.shape) == 2:
                dummy_Z = np.zeros((T_mb.shape[0], 1, settings.get('latent_dim', 100)))
                dummy_G = np.zeros_like(T_mb)
                dummy_errors = np.ones(T_mb.shape[0])
            else:
                dummy_Z = np.zeros((T_mb.shape[0], T_mb.shape[1], settings.get('latent_dim', 100)))
                dummy_G = np.zeros_like(T_mb)
                dummy_errors = np.ones(T_mb.shape[0])
        else:
            dummy_Z = np.zeros((1, 1, settings.get('latent_dim', 100)))
            dummy_G = np.zeros((1, 1, 1))
            dummy_errors = np.ones(1)
        
        return dummy_G, dummy_Z, dummy_errors, 1.0