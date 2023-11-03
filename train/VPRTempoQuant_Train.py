#MIT License

#Copyright (c) 2023 Adam Hines, Peter G Stratton, Michael Milford, Tobias Fischer

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

'''
Imports
'''

import os
import torch
import gc
import sys
sys.path.append('./src')
sys.path.append('./models')
sys.path.append('./output')
sys.path.append('./dataset')

import blitnet as bn
import numpy as np
import torch.nn as nn
import torch.quantization as quantization

from settings import configure, model_logger
from dataset import CustomImageDataset, ProcessImage
from torch.utils.data import DataLoader
from torch.ao.quantization import QuantStub, DeQuantStub
from tqdm import tqdm

class VPRTempoQuant_Train(nn.Module):
    def __init__(self):
        super(VPRTempoQuant_Train, self).__init__()

        # Configure the network
        configure(self)

        # Add quantization stubs for Quantization Aware Training (QAT)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Define the add function for quantized addition
        self.add = nn.quantized.FloatFunctional()      

        # Set the number of expert modules
        self.experts = nn.ModuleList([self.create_modules() for _ in range(self.number_modules)])  

    def create_modules(self):

         # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        module_layers = nn.ModuleDict()

        """
        Define trainable layers here
        """
        feature_layer = bn.SNNLayer(
            dims=[self.input, self.feature],
            thr_range=[0, 0.5],
            fire_rate=[0.2, 0.9],
            ip_rate=0.15,
            stdp_rate=0.001,
            p=[0.1, 0.5]
        )
        
        output_layer = bn.SNNLayer(
            dims=[self.feature, self.output],
            stdp_rate=0.001,
            spk_force=True
        )
        
        module_layers['feature_layer'] = feature_layer
        module_layers['output_layer'] = output_layer

        # Add layer name and index to the layer_dict
        self.layer_dict['feature_layer'] = self.layer_counter
        self.layer_counter += 1  
        self.layer_dict['output_layer'] = self.layer_counter
        self.layer_counter += 1  
        
        return module_layers                       
        
    def model_logger(self):
        """
        Log the model configuration to the console.
        """
        model_logger(self)

    def _anneal_learning_rate(self, layer, mod, itp, stdp):
        """
        Anneal the learning rate for the current layer.
        """
        if np.mod(mod, 100) == 0: # Modify learning rate every 100 timesteps
            pt = pow(float(self.T - mod) / self.T, self.annl_pow)
            layer.eta_ip = torch.mul(itp, pt) # Anneal intrinsic threshold plasticity learning rate
            layer.eta_stdp = torch.mul(stdp, pt) # Anneal STDP learning rate
            
        return layer

    def train_model(self, train_dataset, layer_name, prev_layers=None, model=None, pbar=None):
        """
        Train a module of the network model.

        :param train_dataset: Training dataset
        :param layer_name: Name of the layer to train
        :param prev_layers: Previous layers to pass data through
        :param model: Model containing configurations
        :param pbar: Progress bar (if provided)
        """

        def get_dataloader_for_module(module_index, model):
            # Define the start and end indices for this module
            start_idx = module_index * model.module_images
            end_idx = start_idx + model.module_images

            # Slice the dataset
            subset = torch.utils.data.Subset(train_dataset, indices=range(start_idx, end_idx))

            # Create a DataLoader for the subset
            train_loader = DataLoader(subset, 
                                    batch_size=1, 
                                    shuffle=True,
                                    num_workers=8,
                                    persistent_workers=True)

            return train_loader, start_idx, end_idx

        # Initialize the tqdm progress bar for modules
        pbar_module = tqdm(total=int(len(self.experts)),
                    desc="Training module progress",
                    position=1,
                    colour='RED',
                    ncols=100)

        for module_index, module in enumerate(self.experts):
            layer = module[layer_name]

            init_itp = layer.eta_ip.detach()
            init_stdp = layer.eta_stdp.detach()
            mod = 0  # Used to determine the learning rate annealment

            # Initialize the tqdm progress bar for layers
            pbar_layer = tqdm(total=int(self.T),
                            desc="Training layer progress",
                            position=2,
                            colour='GREEN',
                            leave=True,
                            ncols=100)

            # Get the DataLoader for the current module
            train_loader, start, end = get_dataloader_for_module(module_index, model)
            layer_bounds = []
            for n in reversed(range(1, model.location_repeat)):
                layer_bounds.append(int(end - (n * (model.module_images / model.location_repeat)) - 1))

            # Run training for the specified number of epochs
            for epoch in range(self.epoch):
                for spikes, labels, idx in train_loader:
                    spikes, labels, idx = spikes.to(self.device), labels.to(self.device), idx.to(self.device)
                    
                    # Pass through previous layers if they exist
                    if prev_layers:
                        with torch.no_grad():
                            for prev_layer_name in prev_layers:
                                prev_layer = getattr(module, prev_layer_name)
                                spikes = self.forward(spikes, prev_layer)
                                spikes = bn.clamp_spikes(spikes, prev_layer)
                    else:
                        prev_layer = None

                    # Get the output spikes from the current layer
                    pre_spike = spikes.detach()
                    spikes = self.forward(spikes, layer)
                    spikes_noclp = spikes.detach()
                    spikes = bn.clamp_spikes(spikes, layer)

                    # Calculate STDP
                    layer = bn.calc_stdp(pre_spike, spikes, spikes_noclp, layer, model, module_index, layer_bounds, idx, prev_layer=prev_layer)

                    # Adjust learning rates
                    layer = self._anneal_learning_rate(layer, mod, init_itp, init_stdp)

                    # Update the annealing mod & progress bar 
                    mod += 1
                    pbar_layer.update(1)

            # Close the tqdm progress bar for layers
            pbar_layer.close()
            pbar_module.update(1)

            # Close the tqdm progress bar for modules
        pbar_module.close()


        # Free up memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def forward(self, spikes, layer):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        
        spikes = self.quant(spikes)
        spikes = self.add.add(layer.exc(spikes), layer.inh(spikes))
        spikes = self.dequant(spikes)
        
        return spikes
    
    def save_model(self, model_out):    
        """
        Save the trained model to models output folder.
        """
        torch.save(self.state_dict(), model_out) 
            
def generate_model_name(model):
    """
    Generate the model name based on its parameters.
    """
    return ("VPRTempoQuant" +
            str(model.input) +
            str(model.feature) +
            str(model.output) +
            str(model.number_modules) +
            '.pth')

def check_pretrained_model(model_name):
    """
    Check if a pre-trained model exists and prompt the user to retrain if desired.
    """
    if os.path.exists(os.path.join('./models', model_name)):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt).strip().lower()
        return retrain == 'n'
    return False

def train_new_model(model, model_name, qconfig):
    """
    Train a new model.

    :param model: Model to train
    :param model_name: Name of the model to save after training
    :param qconfig: Quantization configuration
    """
    # Initialize the image transforms and datasets
    image_transform = ProcessImage(model.dims, model.patches)
    train_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                       img_dirs=model.training_dirs,
                                       transform=image_transform,
                                       skip=model.filter,
                                       max_samples=model.number_training_images,
                                       max_samples_per_module=int(model.module_images/model.location_repeat),
                                       test=False)
    
    # Set the model to training mode and move to device
    model.train()
    model.to('cpu')
    model.qconfig = qconfig

    # Apply quantization configurations to the model
    model = quantization.prepare_qat(model, inplace=False)

    # Keep track of trained layers to pass data through them
    trained_layers = [] 

    # Training each layer
    for layer_name, _ in sorted(model.layer_dict.items(), key=lambda item: item[1]):
        print(f"Training layer: {layer_name}")
        # Retrieve the layer object
        #layer = getattr(model, layer_name)
        # Train the layer
        model.train_model(train_dataset, layer_name, prev_layers=trained_layers, model=model)
        # After training the current layer, add it to the list of trained layers
        trained_layers.append(layer_name)
    # Convert the model to eval
    model = quantization.convert(model, inplace=False)
    model.eval()
    # Save the model
    model.save_model(os.path.join('./models', model_name))    

if __name__ == "__main__":
    # Set the number of threads for PyTorch
    torch.set_num_threads(4)
    # Initialize the model
    model = VPRTempoQuant_Train()
    # Set the quantization configuration
    qconfig = quantization.get_default_qat_qconfig('fbgemm')
    # Generate the model name
    model_name = generate_model_name(model)
    # Check if a pre-trained model exists
    use_pretrained = check_pretrained_model(model_name)
    # Train or run inference based on the user's input
    if not use_pretrained:
        train_new_model(model, model_name, qconfig) # Training