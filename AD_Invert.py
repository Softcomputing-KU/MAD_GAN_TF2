import tensorflow as tf
import numpy as np
import json
import os

import data_utils
import plotting
import model
import mmd
import utils
import evaluate
import DR_dis  # Import our modified DR_dis.py module

"""
Here, both the discriminator and generator are used for anomaly detection
Converted to TensorFlow 2.5 style
"""

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: 
    settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
print('Loading data from', data_path)
samples, labels, index = data_utils.get_data(
    settings["data"], settings["seq_length"], settings["seq_step"],
    settings["num_signals"], settings["sub_id"], settings["eval_single"],
    settings["eval_an"], data_path
)

# --- save settings, data --- #
print('Ready to run with settings:')
for (k, v) in settings.items(): 
    print(v, '\t', k)
# add the settings to local environment
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

class myADclass():
    def __init__(self, epoch, settings=settings, samples=samples, labels=labels, index=index):
        """
        Initialize Anomaly Detection class.
        
        Args:
            epoch: Current epoch
            settings: Model settings
            samples: Training samples
            labels: Label data
            index: Index data
        """
        self.epoch = epoch
        self.settings = settings
        self.samples = samples
        self.labels = labels
        self.index = index
        
    def ADfunc(self):
        """Run anomaly detection evaluation and return results."""
        # Calculate number of samples
        num_samples_t = self.samples.shape[0]
        t_size = 500  # Number of test samples to use
        
        # Select random indices for testing
        T_index = np.random.choice(num_samples_t, size=t_size, replace=False)
        print('sample_shape:', self.samples.shape[0])
        print('num_samples_t', num_samples_t)

        # Initialize arrays to store results
        D_test = np.empty([t_size, self.settings['seq_length'], 1])
        DL_test = np.empty([t_size, self.settings['seq_length'], 1])
        GG = np.empty([t_size, self.settings['seq_length'], self.settings['num_signals']])
        T_samples = np.empty([t_size, self.settings['seq_length'], self.settings['num_signals']])
        L_mb = np.empty([t_size, self.settings['seq_length'], 1])
        I_mb = np.empty([t_size, self.settings['seq_length'], 1])
        
        # Process each sample individually
        for batch_idx in range(0, t_size):
            # Display progress
            model.display_batch_progression(batch_idx, t_size)
            
            # Get test sample and corresponding labels
            T_mb = self.samples[T_index[batch_idx], :, :]
            L_mmb = self.labels[T_index[batch_idx], :, :]
            I_mmb = self.index[T_index[batch_idx], :, :]
            
            # Prepare path to model parameters
            para_path = f'./experiments/parameters/{self.settings["sub_id"]}_{self.settings["seq_length"]}_{self.epoch}.npy'
            
            # Run discriminator model
            D_t, L_t = DR_dis.dis_D_model(self.settings, T_mb, para_path)
            
            # Run generator inversion
            Gs, Zs, error_per_sample, heuristic_sigma = DR_dis.invert(
                self.settings, T_mb, para_path,
                g_tolerance=None, e_tolerance=0.1, 
                n_iter=None, max_iter=1000, heuristic_sigma=None
            )
            
            # Handle different output shapes
            # If Gs is 3D (batch, seq_len, features) but expected 2D (seq_len, features)
            if len(Gs.shape) == 3 and GG[batch_idx].ndim == 2:
                if Gs.shape[0] == 1:  # Single sample case
                    Gs = Gs[0]  # Remove batch dimension
                else:
                    Gs = Gs.reshape(self.settings['seq_length'], self.settings['num_signals'])
            
            # If Gs is 2D (batch, features) but expected 3D
            elif len(Gs.shape) == 2 and GG[batch_idx].ndim == 3:
                Gs = Gs.reshape(1, self.settings['seq_length'], self.settings['num_signals'])
            
            # Store results
            GG[batch_idx, :, :] = Gs
            T_samples[batch_idx, :, :] = T_mb
            
            # Handle different shapes of discriminator outputs
            if len(D_t.shape) == 3:
                D_test[batch_idx, :, :] = D_t
            elif len(D_t.shape) == 2:
                D_test[batch_idx, :, :] = D_t.reshape(D_test[batch_idx].shape)
            elif len(D_t.shape) == 1:
                D_test[batch_idx, :, 0] = D_t
            
            if len(L_t.shape) == 3:
                DL_test[batch_idx, :, :] = L_t
            elif len(L_t.shape) == 2:
                DL_test[batch_idx, :, :] = L_t.reshape(DL_test[batch_idx].shape)
            elif len(L_t.shape) == 1:
                DL_test[batch_idx, :, 0] = L_t
            
            L_mb[batch_idx, :, :] = L_mmb
            I_mb[batch_idx, :, :] = I_mmb

        # --- Run anomaly detection evaluations --- #
        results = np.empty([5, 5])
        tao = 0.5  # Threshold value
        lam = 0.8  # Weight parameter
        
        # Method 1: Discriminator logits-based
        Accu1, Pre1, Rec1, F11, FPR1, D_L1 = DR_dis.detection_D_I(
            DL_test, L_mb, I_mb, self.settings['seq_step'], tao
        )
        print(f'seq_length: {self.settings["seq_length"]}')
        print(f'D:Comb-logits-based-Epoch: {self.epoch}; tao={tao:.1f}; '
              f'Accu: {Accu1:.4f}; Pre: {Pre1:.4f}; Rec: {Rec1:.4f}; F1: {F11:.4f}; FPR: {FPR1:.4f}')
        results[0, :] = [Accu1, Pre1, Rec1, F11, FPR1]

        # Method 2: Discriminator statistic-based
        Accu2, Pre2, Rec2, F12, FPR2, D_L2 = DR_dis.detection_D_I(
            D_test, L_mb, I_mb, self.settings['seq_step'], tao
        )
        print(f'seq_length: {self.settings["seq_length"]}')
        print(f'D:Comb-statistic-based-Epoch: {self.epoch}; tao={tao:.1f}; '
              f'Accu: {Accu2:.4f}; Pre: {Pre2:.4f}; Rec: {Rec2:.4f}; F1: {F12:.4f}; FPR: {FPR2:.4f}')
        results[1, :] = [Accu2, Pre2, Rec2, F12, FPR2]

        # Method 3: Combined logits-based
        Accu3, Pre3, Rec3, F13, FPR3, D_L3 = DR_dis.detection_R_D_I(
            DL_test, GG, T_samples, L_mb, self.settings['seq_step'], tao, lam
        )
        print(f'seq_length: {self.settings["seq_length"]}')
        print(f'RD:Comb-logits_based-Epoch: {self.epoch}; tao={tao:.1f}; '
              f'Accu: {Accu3:.4f}; Pre: {Pre3:.4f}; Rec: {Rec3:.4f}; F1: {F13:.4f}; FPR: {FPR3:.4f}')
        results[2, :] = [Accu3, Pre3, Rec3, F13, FPR3]

        # Method 4: Combined statistic-based
        Accu4, Pre4, Rec4, F14, FPR4, D_L4 = DR_dis.detection_R_D_I(
            D_test, GG, T_samples, L_mb, self.settings['seq_step'], tao, lam
        )
        print(f'seq_length: {self.settings["seq_length"]}')
        print(f'RD:Comb-statistic-based-Epoch: {self.epoch}; tao={tao:.1f}; '
              f'Accu: {Accu4:.4f}; Pre: {Pre4:.4f}; Rec: {Rec4:.4f}; F1: {F14:.4f}; FPR: {FPR4:.4f}')
        results[3, :] = [Accu4, Pre4, Rec4, F14, FPR4]

        # Method 5: Reconstruction-based
        Accu5, Pre5, Rec5, F15, FPR5, D_L5 = DR_dis.detection_R_I(
            GG, T_samples, L_mb, self.settings['seq_step'], tao
        )
        print(f'seq_length: {self.settings["seq_length"]}')
        print(f'G:Comb-sample-based-Epoch: {self.epoch}; tao={tao:.1f}; '
              f'Accu: {Accu5:.4f}; Pre: {Pre5:.4f}; Rec: {Rec5:.4f}; F1: {F15:.4f}; FPR: {FPR5:.4f}')
        results[4, :] = [Accu5, Pre5, Rec5, F15, FPR5]

        return results, GG, D_test, DL_test


if __name__ == "__main__":
    print('Main Starting...')
    
    # Create required directories
    os.makedirs('./experiments/plots', exist_ok=True)

    # Initialize result arrays
    Results = np.empty([settings['num_epochs'], 5, 5])
    
    t_size = 500
    D_test = np.empty([settings['num_epochs'], t_size, settings['seq_length'], 1])
    DL_test = np.empty([settings['num_epochs'], t_size, settings['seq_length'], 1])
    GG = np.empty([settings['num_epochs'], t_size, settings['seq_length'], settings['num_signals']])

    # Run for each epoch
    for epoch in range(settings['num_epochs']):
        print(f"\nProcessing epoch {epoch}")
        ob = myADclass(epoch)
        try:
            epoch_results = ob.ADfunc()
            Results[epoch, :, :], GG[epoch, :, :, :], D_test[epoch, :, :, :], DL_test[epoch, :, :, :] = epoch_results
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next epoch

    # Save results
    res_path = f'./experiments/plots/Results_Invert_{settings["sub_id"]}_{settings["seq_length"]}.npy'
    np.save(res_path, Results)

    dg_path = f'./experiments/plots/DG_Invert_{settings["sub_id"]}_{settings["seq_length"]}_'
    np.save(dg_path + 'D_test.npy', D_test)
    np.save(dg_path + 'DL_test.npy', DL_test)
    np.save(dg_path + 'GG.npy', GG)  # Fixed to save GG instead of DL_test again

    print('Main Terminating...')