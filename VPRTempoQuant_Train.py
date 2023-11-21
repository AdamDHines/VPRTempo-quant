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
import math
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
from torch.quantization.observer import MinMaxObserver
from tqdm import tqdm

class VPRTempo(nn.Module):
    def __init__(self):
        super(VPRTempo, self).__init__()

        # Configure the network
        configure(self)

        # Add quantization stubs for Quantization Aware Training (QAT)
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
            thr_range=[0, 0.5],
            fire_rate=[0.2, 0.9],
            ip_rate=0.15,
            stdp_rate=0.005,
            p=[0.1, 0.5],
            device=self.device
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            ip_rate=0.15,
            stdp_rate=0.005,
            spk_force=True,
            device=self.device
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

    def train_model(self, train_loader, layer, prev_layers=None):
        """
        Train a layer of the network model.

        :param train_loader: Training data loader
        :param layer: Layer to train
        :param prev_layers: Previous layers to pass data through
        """

        # Initialize the tqdm progress bar
        pbar = tqdm(total=int(self.T),
                    desc="Training ",
                    position=0)
        
        # Initialize the learning rates for each layer (used for annealment)
        init_itp = layer.eta_ip.detach()
        init_stdp = layer.eta_stdp.detach()
        mod = 0  # Used to determine the learning rate annealment, resets at each epoch
        # Run training for the specified number of epochs
        for epoch in range(self.epoch):
            # Run training for the specified number of timesteps
            for spikes, labels in train_loader:
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                idx = labels / self.filter # Set output index for spike forcing
                # Pass through previous layers if they exist
                if prev_layers:
                    with torch.no_grad():
                        for prev_layer_name in prev_layers:
                            prev_layer = getattr(self, prev_layer_name) # Get the previous layer object
                            spikes = self.forward(spikes, prev_layer) # Pass spikes through the previous layer
                            spikes = bn.clamp_spikes(spikes, prev_layer) # Clamp spikes [0, 0.9]
                else:
                    prev_layer = None
                # Get the output spikes from the current layer
                pre_spike = spikes.detach() # Previous layer spikes for STDP
                spikes = self.forward(spikes, layer) # Current layer spikes
                spikes_noclp = spikes.detach() # Used for inhibitory homeostasis
                spikes = bn.clamp_spikes(spikes, layer) # Clamp spikes [0, 0.9]
                # Calculate STDP
                layer = bn.calc_stdp(pre_spike,spikes,spikes_noclp,layer, idx, prev_layer=prev_layer)
                # Adjust learning rates
                layer = self._anneal_learning_rate(layer, mod, init_itp, init_stdp)
                # Update the annealing mod & progress bar 
                mod += 1
                pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()

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

class PowerOfTwoMinMaxObserver(MinMaxObserver):
    """
    Observer module for computing the quantization parameters based on the
    running min and max values, with scales as powers of two.

    This observer extends the MinMaxObserver to use scales that are powers of two.
    It overrides the calculate_qparams method to compute the power of two scale.
    """

    def calculate_qparams(self):
        r"""Calculates the quantization parameters with scale as a power of two."""
        min_val, max_val = self.min_val.item(), self.max_val.item()

        # Calculate the scale as the nearest power of two
        max_range = max(abs(min_val), abs(max_val))
        scale = 2 ** math.ceil(math.log2(max_range / (self.quant_max - self.quant_min)))

        # Calculate zero_point as in the base class
        if self.qscheme == torch.per_tensor_symmetric:
            if self.dtype == torch.qint8:
                zero_point = 0
            else:
                zero_point = 128
        else:
            zero_point = self.quant_min - round(min_val / scale)
            # Clamp zero_point to be within 0-255
            zero_point = max(0, min(zero_point, 255))

        # Convert scale and zero_point to PyTorch tensors
        scale = torch.tensor(scale, dtype=torch.float32)
        zero_point = torch.tensor(zero_point, dtype=torch.int64)

        return scale, zero_point


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
                                       test=False)
    # Initialize the data loader
    train_loader = DataLoader(train_dataset, 
                              batch_size=1, 
                              shuffle=True,
                              num_workers=8,
                              persistent_workers=True)
    # Set the model to training mode and move to device
    model.train()
    model.to('cpu')
    # Set the quantization configuration with the custom observer for activations
    qconfig = quantization.get_default_qat_qconfig('fbgemm')
    custom_activation_observer = PowerOfTwoMinMaxObserver.with_args()  # Create an instance of your custom observer
    # Set the custom observer for activations in the qconfig
    qconfig = torch.quantization.default_qconfig._replace(activation=custom_activation_observer)

    model.qconfig = qconfig

    # Apply quantization configurations to the model
    model = quantization.prepare_qat(model, inplace=False)

    # Set the custom weight observer for weights in the qconfig
    #custom_weight_observer = PowerOfTwoWeightObserver.with_args()
    #model.qconfig = model.qconfig._replace(weight=custom_weight_observer)

    # Keep track of trained layers to pass data through them
    trained_layers = [] 

    # Training each layer
    for layer_name, _ in sorted(model.layer_dict.items(), key=lambda item: item[1]):
        print(f"Training layer: {layer_name}")
        # Retrieve the layer object
        layer = getattr(model, layer_name)
        # Train the layer
        model.train_model(train_loader, layer, prev_layers=trained_layers)
        # After training the current layer, add it to the list of trained layers
        trained_layers.append(layer_name)
    # Convert the model to a quantized model
    model = quantization.convert(model, inplace=False)
    model.eval()
    # Save the model
    model.save_model(os.path.join('./models', model_name))    

if __name__ == "__main__":
    # Set the number of threads for PyTorch
    torch.set_num_threads(8)
    # Initialize the model
    model = VPRTempo()
    # Initialize the logger
    model.model_logger()
    # Set the quantization configuration
    if model.quantize:
        qconfig = quantization.get_default_qat_qconfig('fbgemm')
    else:
        raise ValueError("Quantization must be enabled for training.")
    # Generate the model name
    model_name = generate_model_name(model)
    # Check if a pre-trained model exists
    use_pretrained = check_pretrained_model(model_name)
    # Train or run inference based on the user's input
    if not use_pretrained:
        train_new_model(model, model_name, qconfig) # Training
    model.logger.info('Training complete.')