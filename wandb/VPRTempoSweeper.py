import torch
import pprint
import wandb
import tqdm
import gc
import tracemalloc
import sys
sys.path.append('.')
sys.path.append('./src')

import torch.nn as nn
import torch.quantization as quantization
import matplotlib.pyplot as plt
import numpy as np

from VPRTempo import VPRTempo
from VPRTempoQuant import VPRTempoQuant
from settings import configure, image_csv
from dataset import ProcessImage, CustomImageDataset
from torch.utils.data import DataLoader

class VPRTempoWand(nn.Module):
    def __init__(self, fl, fh, n_init, n_itp, p_exc, p_inh, theta_max,
                 quant=False):
        super(VPRTempoQuant, self).__init__()

        # Configure the network
        configure(self)
        
        # Define the images to load (both training and inference)
        image_csv(self)

        # Add quantization stubs for Quantization Aware Training (QAT)
        if quant:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
            # Define the add function for quantized addition
            self.add = nn.quantized.FloatFunctional()      

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        """
        Define trainable layers here
        """
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            thr_range=[0, theta_max],
            fire_rate=[fl, fh],
            ip_rate=n_itp,
            stdp_rate=n_init,
            const_inp=[0, 0.0],
            p=[p_exc, p_inh]
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            ip_rate=n_itp,
            stdp_rate=n_init,
            spk_force=True
        )
        
    def add_layer(self, name, **kwargs):
        """
        Dynamically add a layer with given name and keyword arguments.
        
        :param name: Name of the layer to be added
        :type name: str
        :param kwargs: Hyperparameters for the layer
        """
        # Check for layer name duplicates
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        
        # Add a new SNNLayer with provided kwargs
        setattr(self, name, bn.SNNLayer(**kwargs))
        
        # Add layer name and index to the layer_dict
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1       

class VPRTempoSweeper():
    def __init__(self):
        super(VPRTempoSweeper, self).__init__()

        # Configure the network parameters
        configure(self)
        self.quantize = False # Run True to run VPRTempoQuant instead of base VPRTempo
        self.wandb = False # Run True to log to wandb
        self.num_runs = 1 # Number of runs to perform (for both wandb and normal)

        # Initialize the image transforms and datasets
        self.image_transform = ProcessImage(self.dims, self.patches)
        self.train_dataset = CustomImageDataset(annotations_file=self.dataset_file, 
                                        img_dirs=self.training_dirs,
                                        transform=self.image_transform,
                                        skip=self.filter,
                                        max_samples=self.number_training_images,
                                        test=False)
        self.test_dataset = CustomImageDataset(annotations_file=self.dataset_file, 
                                      img_dirs=self.testing_dirs,
                                      transform=self.image_transform,
                                      skip=self.filter,
                                      max_samples=self.number_testing_images)
        # Initialize the data loaders
        self.train_loader = DataLoader(self.train_dataset, 
                                batch_size=1, 
                                shuffle=False,
                                num_workers=2,
                                persistent_workers=True)
        self.test_loader = DataLoader(self.test_dataset, 
                                batch_size=1, 
                                shuffle=False,
                                num_workers=2,
                                persistent_workers=True)
        
    def new_model(self):
        # Define a new network model
        if not self.quantize:
            model = VPRTempo()
        else:
            model = VPRTempoQuant()

        return model
    
    def wandb_run(self):
        wandb.login()
        # define the method and parameters for grid search
        sweep_config = {'method':'random'}
        metric = {'name':'p100r', 'goal':'maximize'}
        
        sweep_config['metric'] = metric
        
        parameters_dict = {
                'f_rateH': {
                    'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                },
                'f_rateL': {
                    'values': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
                },
                'n_init': {
                    'values': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
                },
                'n_itp': {
                    'values': [0.025, 0.0812, 0.1375, 0.1938, 0.25]
                },
                'p_exc': {
                    'values': [0.01, 0.04, 0.06, 0.09, 0.12, 0.14, 0.17, 0.2, 0.22, 0.25]
                },
                'p_inh': {
                    'values': [0.1, 0.17, 0.24, 0.32, 0.39, 0.46, 0.53, 0.61, 0.68, 0.75]
                },
                'theta_max': {
                    'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                }
            } 
        
        sweep_config['parameters'] = parameters_dict
        pprint.pprint(sweep_config)
        
        # start sweep controller
        sweep_id = wandb.sweep(sweep_config, project="vprtempo-grid-search")

        # Create wandb model
        model = VPRTempoWand()

        # Define the operational model based on quantization
        if self.quantize:
            op_model = VPRTempoQuant()
        else:
            op_model = VPRTempo()  

        

        def wandsearch(config=None):
            with wandb.init(config=config):
                config = wandb.config
                model.train(config.f_rateH,
                            config.f_rateL,
                            config.n_init,
                            config.n_itp,
                            config.p_exc,
                            config.p_inh,
                            config.theta_max)
                
                p100r = model.networktester()
                wandb.log({"p100r" : p100r})
            
        wandb.agent(sweep_id,wandsearch)

    def multi_run(self):
        tracemalloc.start()
        pbar_sweep = tqdm.tqdm(total=self.num_runs,
                         desc="Iterating through VPRTempo runs",
                         position=0)
        
         # If quantizing, set the quantization parameters
        if self.quantize:
            qconfig = quantization.get_default_qat_qconfig('fbgemm')

        # Define outputs to keep track of over iterations
        numcorr_all = []
        idxcorr_all = []

        # Run VPRTempo for specified number of trials
        for trials in range(self.num_runs):
            # Initialize the model
            model = self.new_model()
            model.train()
            # If quantizing, set the quantization configuration
            if self.quantize:
                model.qconfig = qconfig
                model.to('cpu')
                # Apply quantization configurations to the model
                model = quantization.prepare_qat(model, inplace=False)
            # Keep track of trained layers to pass data through them
            trained_layers = [] 

            # Training each layer
            for layer_name, _ in sorted(model.layer_dict.items(), key=lambda item: item[1]):
                print(f"Training layer: {layer_name}")
                # Retrieve the layer object
                layer = getattr(model, layer_name)
                # Train the layer
                model.train_model(self.train_loader, layer, prev_layers=trained_layers)
                # After training the current layer, add it to the list of trained layers
                trained_layers.append(layer_name)

            # If quantizing, convert the model
            if self.quantize:
                model = quantization.convert(model, inplace=False)

            # Inferencing the model
            # Set the model to eval mode
            model.eval()

            # Retrieve layer names for inference
            layer_names = list(model.layer_dict.keys())

            # Use evaluate method for inference accuracy
            numcorr, idxcorr = model.evaluate(self.test_loader, layers=layer_names)
            numcorr_all.append(numcorr)
            idxcorr_all.append(list(idxcorr))

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            for stat in top_stats[:10]:
                print(stat)

            print('')
            pbar_sweep.update(1)
            print('')

            del model
            gc.collect()

        return numcorr_all, idxcorr_all
    
    def plot_dist(self,idxcorr):
        # Flatten the idxcorr list of lists into a single list
        all_results = [item for sublist in idxcorr for item in sublist]

        # Calculate the distribution
        distribution = [all_results.count(i) for i in range(self.number_testing_images)]
        num_trials = len(idxcorr)
        average_distribution = [val/num_trials for val in distribution]

        # Improved plot aesthetics
        plt.figure(figsize=(14, 7))

        # Use a color gradient
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(average_distribution), vmax=max(average_distribution))

        x = range(self.number_testing_images)

        # Using the overall average gradient value for the fill color
        fill_color = cmap(norm(np.mean(average_distribution)))
        plt.fill_between(x, average_distribution, color=fill_color, alpha=0.6)

        # Plot each segment of the curve with its color
        for i in range(len(x) - 1):
            segment_color = cmap(norm(average_distribution[i]))
            plt.plot([x[i], x[i+1]], [average_distribution[i], average_distribution[i+1]], color=segment_color, lw=2)

        # Adding a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, orientation='vertical', fraction=0.03, pad=0.05)
        cbar.set_label('Average Occurrence', rotation=270, labelpad=20)

        plt.title("Average Distribution of 'Correct' Values")
        plt.xlabel('Value')
        plt.ylabel('Average Occurrence')

        # Add margins for the axes
        x_margin = 0.05 * self.number_testing_images
        y_margin = 0.05 * max(average_distribution)
        plt.xlim(0 - x_margin, self.number_testing_images + x_margin)
        plt.ylim(0, max(average_distribution) + y_margin)

        plt.tight_layout()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()  

if __name__ == "__main__":
    # Initialize sweeper and VPRTempo model
    sweeper = VPRTempoSweeper()
    torch.set_num_threads(2)
    # If running wandb, initialize and run wandb
    if sweeper.wandb:
        sweeper.wandb_run()
    # Otherwise, run normal sweeps and collect output
    else:
        numcorr, idxcorr = sweeper.multi_run()
        print(numcorr)
    sweeper.plot_dist(idxcorr)