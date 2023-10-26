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

# Get the 2D patches or the patch normalization

'''
Imports
'''
import cv2
import os
import math
import torch

import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import blitnet as bn
import torch.nn as nn

from settings import configure, image_csv
from dataset import CustomImageDataset, ProcessImage
from torch.utils.data import DataLoader
from metrics import recallAtK, createPR, recallAt100precision
from timeit import default_timer
from os import path

class validate(nn.Module):
    def __init__(self, out, quant=False):
        super(validate, self).__init__()

        configure(self)
        image_csv(self)

        self.out = out
        self.quant = quant

    # plot similarity matrices
    def plot_similarity(self, mat, name, cmap, ax=None, dpi=600):
        
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi,figsize=(8, 6))
        else:
            fig = ax.get_figure()

        cax = ax.matshow(mat, cmap=cmap, aspect='equal')
        fig.colorbar(cax, ax=ax, label="Spike amplitude")
        ax.set_title(name,fontsize = 12)
        ax.set_xlabel("Query",fontsize = 12)
        ax.set_ylabel("Database",fontsize = 12)
        
    # plot weight matrices
    def plot_weights(W, name, cmap, vmax, dims, ax=None):
        newx = dims[0]
        newy = dims[1]
        
        # loop through expert modules and output weights
        init_weight = np.array([])
        for n in range(len(W[:,0,0])):
            init_weight = np.append(init_weight, np.reshape(W[n,:,:].cpu().numpy(), (dims[2],)))
        
        # reshape the weight matrices
        reshape_weight = np.reshape(init_weight, (newx, newy))
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # plot the weight matrix to specified subplot
        cax = ax.matshow(reshape_weight, cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(cax, ax=ax, label="Weight strength",shrink=0.5)
        
        # set figure titles and labels
        ax.set_title(name, fontsize = 12)
        ax.set_xlabel("x-weights", fontsize = 12)
        ax.set_ylabel("y-weights", fontsize = 12)

    # plot PR curves
    def plot_PR(self,P, R, name, ax=None):

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(R, P)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)

    # plot the recall@N
    def plot_recallN(self,recallN, N_vals, name, ax=None):
        
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        
            ax.plot(N_vals, recallN)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("N", fontsize=12)
            ax.set_ylabel("Recall", fontsize=12)
        
    # run recallAtK() function from VPR Tutorial
    def recallAtN(self,S_in, GThard, GTsoft, N):
        
        # run recall at N over each value of N
        recall_list = []
        for n in N:
            recall_list.append(recallAtK(S_in, GThard, GTsoft, K=n))
            
        return recall_list
        
    def sad(fullTrainPaths, filteredNames, imWidth, imHeight, num_patches, testPath, 
            test_location, imgs, ids, number_testing_images, number_training_images,
            validation):

        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Setting up Sum of Absolute Differences (SAD) calculations')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')

        sadcorrect = 0
        # load the training images
        imgs['training'], ids['training'] = ut.loadImages(False, 
                                                        fullTrainPaths, 
                                                        filteredNames, 
                                                        [imWidth,imHeight], 
                                                        num_patches, 
                                                        testPath, 
                                                        test_location)
        del imgs['training'][number_training_images:len(imgs['training'])]
        
        # create database tensor
        for ndx, n in enumerate(imgs['training']):
            if ndx == 0:
                db = torch.unsqueeze(n,0)
            else:
                db = torch.cat((db, torch.unsqueeze(n,0)), 0)

        def calc_sad(query, database, const):
            SAD = torch.sum(torch.abs((database * const) - (query * const)), (1,2), keepdim=True)
            for n in range(2):
                SAD = torch.squeeze(SAD,-1)
            return SAD

        # calculate SAD for each image to database and count correct number
        imgred = 1/(imWidth * imHeight)
        sad_concat = []

        print('Running SAD')
        correctidx = []
        incorrectidx = []

        start = default_timer()
        for n, q in enumerate(imgs['testing']):
            pixels = torch.empty([])

            # create 3D tensor of query images
            for o in range(number_testing_images):
                if o == 0:
                    pixels = torch.unsqueeze(q,0)
                else:
                    pixels = torch.cat((pixels,torch.unsqueeze(q,0)),0)
            
            sad_score = calc_sad(pixels, db, imgred)
            best_match = np.argmin(sad_score.cpu().numpy())

            if n == best_match:
                sadcorrect += 1
                correctidx.append(n)
            else:
                incorrectidx.append(n)

            if validation:
                sad_concat.append(sad_score.cpu().numpy())

        end = default_timer()   

        p100r_local = round((sadcorrect/number_testing_images)*100,2)
        print('')
        print('Sum of absolute differences P@1: '+ str(p100r_local) + '%')
        print('Sum of absolute differences queried at ' + str(round(number_testing_images/(end-start),2)) + 'Hz')
        
        GT = np.zeros((number_testing_images,number_training_images), dtype=int)
        for n in range(len(GT)):
            GT[n,n] = 1
        sad_concat = (1-np.reshape(np.array(sad_concat),(number_training_images,number_testing_images)))
        P,R  = createPR(sad_concat,GT,GT,matching="single")
        for n, ndx in enumerate(P):
            P[n] = round(ndx,2)
            R[n] = round(R[n],2)

        # make the PR curve
        fig = plt.figure()
        plt.plot(R,P)
        fig.suptitle("Precision Recall curve",fontsize = 12)
        plt.xlabel("Recall",fontsize = 12)
        plt.ylabel("Precision",fontsize = 12)
        plt.show()
        
        # calculate the recall at N
        N_vals = [1,5,10,15,20,25]
        recallN = ut.recallAtN(sad_concat, GT, GT, N_vals)
        
        return P,R,recallN,N_vals

    # clear the contents of the weights folder if retraining with same settings
    def clear_weights(training_out):
        if os.path.isfile(training_out + 'net.pkl'):
            os.remove(training_out+'net.pkl')
        if os.path.isfile(training_out + 'GT_imgnames.pkl'):
            os.remove(training_out+'GT_imgnames.pkl')
        if not os.path.isdir(training_out):
            os.mkdir(training_out)

    def load_model(self, model_path):
        """
        Load pre-trained model and set the state dictionary keys.
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device),
                             strict=True)  

    def evaluate(self, model, test_loader, layers=None):
        """
        Run the inferencing model and calculate the accuracy.

        :param test_loader: Testing data loader
        :param layers: Layers to pass data through
        """
        out = []

        # Run inference for the specified number of timesteps
        for spikes, labels, idx in test_loader:
            # Set device
            spikes, labels, idx = spikes.to(self.device), labels.to(self.device), idx.to(self.device)
            init_spikes = spikes.detach()
            outputs = []

            # Pass the spikes through each module in the experts
            for module in model.experts:
                if layers:
                    for layer_name in layers:
                        layer = getattr(module, layer_name)
                        spikes = self.calc(spikes, layer)
                        spikes = bn.clamp_spikes(spikes, layer)
                outputs.append(spikes.view(-1))  # Flatten and append to outputs
                spikes = init_spikes.detach()
            
            # Now, gather the outputs to determine the argmax
            out_ten = torch.cat(outputs)
            out.extend(out_ten.cpu().tolist())

        # Reshape into the similarity matrix
        out = np.array(out).T
        out = np.reshape(out, (self.number_training_images, self.number_testing_images))

        return out


    def calc(self, spikes, layer):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        
        spikes = layer.exc(spikes) + layer.inh(spikes)

        return spikes

    def calc_quant(self, spikes, layer, model):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        
        spikes = model.quant(spikes)
        spikes = model.add.add(layer.exc(spikes), layer.inh(spikes))
        spikes = model.dequant(spikes)
        
        return spikes      

    def run_inference(self):
        """
        Run inference on a pre-trained model.

        :param model: Model to run inference on
        :param model_name: Name of the model to load
        :param qconfig: Quantization configuration
        """
        # Set the model
        model = self.model

        # Initialize the image transforms and datasets
        image_transform = ProcessImage(model.dims, model.patches)
        test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                        img_dirs=model.testing_dirs,
                                        transform=image_transform,
                                        skip=model.filter,
                                        max_samples=model.number_testing_images,
                                        max_samples_per_module=int(model.module_images/model.location_repeat))
        # Initialize the data loader
        test_loader = DataLoader(test_dataset, 
                                batch_size=1, 
                                shuffle=False,
                                num_workers=8,
                                persistent_workers=True)

        # Retrieve layer names for inference
        layer_names = list(model.layer_dict.keys())

        # Use evaluate method for inference accuracy
        out = self.evaluate(model, test_loader, layers=layer_names)

        return out

    def forward(self):
        #out = self.run_inference()
         # generate the ground truth matrix
        GT = np.zeros((self.number_testing_images, self.number_training_images), dtype=int)
        for n in range(len(GT)):
            GT[n,n] = 1

        # create the main figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Network metrics', fontsize = 18)
        cmap = plt.cm.tab20c
        # plot the similarity matrices
        self.plot_similarity(self.out, 'VPRTempo similarity', cmap, ax=axes[0,0])
        self.plot_similarity(GT, 'Ground truth', cmap, ax=axes[0,1])

        # get the P & R 
        P, R = createPR(self.out.astype(float), GT, GT, matching="single")
        for n, ndx in enumerate(P):
            P[n] = round(ndx,2)
            R[n] = round(R[n],2)

         # plot the PR curve
        self.plot_PR(P, R, 'Precision-recall curve',ax=axes[1,0])
        print(P)
        print(R)
        # calculate the recall at N
        N_vals = [1, 5, 10, 15, 20, 25]
        recallN = self.recallAtN(self.out.astype(float), GT, GT, N_vals)
        print(recallN)
        # plot the recall at N
        self.plot_recallN(recallN, N_vals, 'Recall@N',ax=axes[1,1])

        plt.tight_layout()
        plt.show()

        R = recallAt100precision(self.out.astype(float), GT, GT, matching='single')
        print(R)

        #fig.savefig(self.model.output_folder+'/metrics.pdf', format='pdf', dpi=300)