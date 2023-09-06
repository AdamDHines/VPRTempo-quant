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

import pickle
import os
import torch
import gc
import timeit
import shutil
import sys
sys.path.append('./src')
sys.path.append('./weights')
sys.path.append('./settings')
sys.path.append('./output')

import blitnet as bn
import utils as ut
import numpy as np
import matplotlib.pyplot as plt

from os import path
from metrics import createPR
from datetime import datetime


'''
Spiking network model class
'''
class snn_model():
    def __init__(self):
        super().__init__()
        
        '''
        USER SETTINGS
        '''
        self.dataset = 'nordland' # set which dataset to run network on
        self.trainingPath = '/home/adam/data/nordland/' # training datapath
        self.testPath = '/home/adam/data/nordland/'  # testing datapath
        self.number_modules = 1 # number of module networks
        self.number_training_images = 100 # Alter number of training images
        self.number_testing_images = 100 # Alter number of testing images
        self.locations = ["fall","spring"] # Define the datasets used in the training
        self.test_location = "summer" # Define the dataset is used for testing
        self.filter = 8 # Set to number of images to filter
        self.validation = True # Set to True to calculate PR metrics
        
        assert (len(self.dataset) != 0),"Dataset not defined, see README.md for details on setting up images"
        assert (os.path.isdir(self.trainingPath)),"Training path not set or path does not exist, edit line 60"
        assert (os.path.isdir(self.testPath)),"Test path not set or path does not exist, edit line 61"
        assert (os.path.isdir(self.trainingPath+self.locations[0])),"Images must be organized into folders based on locations, see README.md for details"
        assert (os.path.isdir(self.testPath+self.test_location)),"Images must be organized into folders based on locations, see README.md for details"
        
        '''
        NETWORK SETTINGS
        '''
        # Image and patch normalization settings
        self.imWidth = 28 # image width for patch norm
        self.imHeight = 28 # image height for patch norm
        self.num_patches = 7 # number of patches
        self.intensity = 255 # divide pixel values to get spikes in range [0,1]
        self.location_repeat = len(self.locations) # Number of training locations that are the same
        
        # Network and training settings
        self.input_layer = (self.imWidth*self.imHeight) # number of input layer neurons
        self.feature_layer = int(self.input_layer*2) # number of feature layer neurons
        self.output_layer = int(self.number_training_images/self.number_modules) # number of output layer neurons
        self.epoch = 4 # number of training iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda": # clear cuda cache, initialize, and syncgronize gpu
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.cuda.set_device(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.init()
            torch.cuda.synchronize(device=self.device)
        self.T = int((self.number_training_images/self.number_modules)
                             *self.location_repeat) # number of training steps
        self.annl_pow = 2 # learning rate anneal power
        self.imgs = {'training':[],'testing':[]}
        self.ids = {'training':[],'testing':[]}
        self.spike_rates = {'training':[],'testing':[]}
        
        # Hyperparamters
        self.theta_max = 0.5 # maximum threshold value
        self.n_init = 0.005 # initial learning rate value
        self.n_itp = 0.15 # initial intrinsic threshold plasticity rate[[0.9999]]
        self.f_rate = [0.2,0.9] # firing rate range
        self.p_exc = 0.1 # probability of excitatory connection
        self.p_inh = 0.5 # probability of inhibitory connection
        self.c= 0.1 # constant input 
        
        # Test settings 
        self.test_true = False # leave default to False
    
        
        '''
        DATA SETTINGS
        '''
        # Select training images from list
        with open('./'+self.dataset+'_imageNames.txt') as file:
            self.imageNames = [line.rstrip() for line in file]
            
        # Filter the loading images based on self.filter
        self.filteredNames = []
        for n in range(0,len(self.imageNames),self.filter):
            self.filteredNames.append(self.imageNames[n])
        
        del self.filteredNames[self.number_training_images:len(self.filteredNames)]

        # Get the full training and testing data paths    
        self.fullTrainPaths = []
        for n in self.locations:
                self.fullTrainPaths.append(self.trainingPath+n+'/')
        
        # Print network details
        print('////////////')
        print('VPRTempo - Temporally Encoded Visual Place Recognition v1.1.0-alpha')
        print('Queensland University of Technology, Centre for Robotics')
        print('')
        print('© 2023 Adam D Hines, Peter G Stratton, Michael Milford, Tobias Fischer')
        print('MIT license - https://github.com/QVPR/VPRTempo')
        print('\\\\\\\\\\\\\\\\\\\\\\\\')
        print('')
        print('CUDA available: '+str(torch.cuda.is_available()))
        if torch.cuda.is_available() == True:
            current_device = torch.cuda.current_device()
            print('Current device is: '+str(torch.cuda.get_device_name(current_device)))
        else:
            print('Current device is: CPU')
 
        # Network weights name
        self.training_out = './weights/'+str(self.input_layer)+'i'+\
                                            str(self.feature_layer)+\
                                        'f'+str(self.output_layer)+\
                                            'o'+str(self.epoch)+'/'
        
        # create output folder
        now = datetime.now()
        self.output_folder = './output/'+now.strftime("%d%m%y-%H-%M-%S")
        os.mkdir(self.output_folder)
    
        
    # Check if pre-trained network exists, prompt if retrain or run        
    def checkTrainTest(self):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        print('')
        if path.isdir(self.training_out):
            retrain = input(prompt)
        else:
            retrain = 'y'
        return retrain
    
    def initialize(self,condition):
        
        '''
        Network startup and initialization
        '''
        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(condition+' startup and initialization')    
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')
        print('Loading '+condition+' images')
        
        if condition == 'testing':
            self.test_true = True
            #random.shuffle(self.filteredNames)
            del self.filteredNames[self.number_testing_images:len(self.filteredNames)]
            self.epoch = 1 # Only run the network once
            self.location_repeat = 1 # One location repeat for testing
            imgNum = self.number_testing_images
        else:
            imgNum = self.number_training_images
            
        # load the training images
        self.imgs[condition], self.ids[condition] = ut.loadImages(self.test_true,
                                            self.fullTrainPaths,
                                            self.filteredNames,
                                            [self.imWidth,self.imHeight],
                                            self.num_patches,
                                            self.testPath,
                                            self.test_location)
        
        self.spike_rates[condition] = ut.setSpikeRates(self.imgs[condition],
                                           self.ids[condition],
                                            self.device,
                                            [self.imWidth,self.imHeight],
                                            self.test_true,
                                            imgNum,
                                            self.number_modules,
                                            self.intensity,
                                            self.location_repeat)
    
    '''
    Run the training network
    '''
    def train(self):

        # remove contents of the weights folder
        if os.path.isfile(self.training_out + 'net.pkl'):
            os.remove(self.training_out+'net.pkl')
        if os.path.isfile(self.training_out + 'GT_imgnames.pkl'):
            os.remove(self.training_out+'GT_imgnames.pkl')
        if os.path.isdir(self.training_out+'images/training/'):
            shutil.rmtree(self.training_out+'images/training/')
        if not os.path.isdir(self.training_out):
            os.mkdir(self.training_out)
        '''
        Network startup and initialization
        '''
        
        print('Creating network and setting weights')
        # create a new blitnet netowrk
        net = bn.newNet(self.number_modules,self.imWidth*self.imHeight)
        
        # add the input layer
        bn.addLayer(net,[self.number_modules,1,self.input_layer],
                             0.0,0.0,0.0,0.0,0.0,False)   
        
        # add the feature layer
        bn.addLayer(net,[self.number_modules,1,self.feature_layer],
                        [0,self.theta_max],
                        [self.f_rate[0],self.f_rate[1]],
                         self.n_itp,
                        [0,self.c],
                         0,
                         False)
        
        # sequentially set the feature firing rates for the feature layer
        fstep = (self.f_rate[1]-self.f_rate[0])/self.feature_layer
        
        # loop through all modules and feature layer neurons
        for x in range(self.number_modules):
            for i in range(self.feature_layer):
                net['fire_rate'][1][x][:,i] = self.f_rate[0]+fstep*(i+1)
            
        # add excitatory inhibitory connections for input and feature layer
        bn.addWeights(net,0,1,[-1,0,1],[self.p_exc,self.p_inh],self.n_init)
        self.init_weights = [net['W'][0].clone().detach(),net['W'][1].clone().detach()]

        '''
        Feature layer training
        '''
        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Training the input to feature layer')    
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')
        
        print('Setting spike rates from loaded images')
        
        # begin timer for network training
        start = timeit.default_timer()
        # Set the spikes times for the input images
        net['set_spks'][0] = self.spike_rates['training']
        layers = [len(net['W'])-2, len(net['W'])-1, len(net['W_lyr'])-1]
        # Train the input to feature layer
        # Train the feature layer
        for epoch in range(self.epoch):
            net['step_num'] = 0
            epochStart = timeit.default_timer()
            for t in range(int(self.T)):
                bn.runSim(net,1,self.device,layers)
                torch.cuda.empty_cache()
                # anneal learning rates
                if np.mod(t,10)==0:
                    pt = pow(float(self.T-t)/self.T,self.annl_pow)
                    net['eta_ip'][1] = self.n_itp*pt
                    net['eta_stdp'][0] = self.n_init*pt
                    net['eta_stdp'][1] = -1*self.n_init*pt
            print('Epoch '+str(epoch+1)+' trained in: '
                  +str(round(timeit.default_timer()-epochStart,2))+'s')
            print('')
            
        print('Finished training input to feature layer')
        
        # delete the training images
        
        '''
        Preparations for feature to output layer training
        '''
        
        # Turn off learning between input and feature layer
        net['eta_ip'][1] = 0.0
        if self.p_exc > 0.0: net['eta_stdp'][0] = 0.0
        if self.p_inh > 0.0: net['eta_stdp'][1] = 0.0
        
        print('Getting feature layer spikes for output layer training')
        # get the feature spikes for training the output layer
        net['x_feat'] = []
        net['step_num'] = 0
        for t in range(int(self.T)): # run blitnet without learning to get feature spikes
            bn.runSim(net,1,self.device,layers)
            net['x_feat'].append(net['x'][1]) # dictionary output of feature spikes
            
        print('Creating output layer')    
        # Create and train the output layer with the feature layer
        bn.addLayer(net,[self.number_modules,1,self.output_layer],
                    0.0,0.0,0.0,0.0,0.0,False)

        # Add excitatory and inhibitory connections
        bn.addWeights(net,1,2,[-1.0,0.0,1.0],[1.0,1.0],self.n_init)
        self.init_weights.append(net['W'][2].clone().detach())
        self.init_weights.append(net['W'][3].clone().detach())
        
        # Output spikes for spike forcing (final layer)
        out_spks = torch.tensor([0],device=self.device,dtype=float)

        '''
        Output layer training
        '''
        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Training the feature to output layer')    
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')
        
        net['spike_dims'] = 1 # change spike dims for output spike indexing
        net['set_spks'][0] = [] # remove input spikes
        layers = [len(net['W'])-2, len(net['W'])-1, len(net['W_lyr'])-1]
        
        # Train the feature to output layer   
        for epoch in range(self.epoch):
            net['step_num'] = 0
            epochStart = timeit.default_timer()
            for t in range(self.T):
                out_spks = torch.tensor([0],device=self.device,dtype=float)

                net['set_spks'][-1] = torch.tile(out_spks,
                                                 (self.number_modules,
                                                  1,
                                                  self.output_layer))
                net['x'][1] = net['x_feat'][t]
                bn.runSim(net,1,self.device,layers)
                # Anneal learning rates
                if np.mod(t,10)==0:
                    pt = pow(float(self.T-t)/(self.T),self.annl_pow)
                    net['eta_ip'][2] = self.n_itp*pt
                    net['eta_stdp'][2] = self.n_init*pt
                    net['eta_stdp'][3] = -1*self.n_init*pt
                if np.mod((t+1),(int(self.T/self.location_repeat))) == 0:
                    net['step_num'] = 0     
            print('Epoch '+str(epoch+1)+' trained in: '
                  +str(round(timeit.default_timer()-epochStart,2))+'s')
            print('')
            
        print('Finished training feature to output layer')
        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Network trained in '+str(round(timeit.default_timer()-start,2))
                                              +'s')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')
        
        # Turn off learning
        net['eta_ip'][2] = 0.0
        net['eta_stdp'][2] = 0.0
        net['eta_stdp'][3] = 0.0

        # Clear the network output spikes
        net['set_spks'][-1] = []
        net['spike_dims'] = self.input_layer
        # Reset network details
        net['sspk_idx'] = [0,0,0]
        net['step_num'] = 0
        net['spikes'] = [[],[],[]]
        net['x'] = [[],[],[]]
        net['x_feat'] = []
        
            
        print('Network formatting and saving...') 
        
        # Output the trained network
        outputPkl = self.training_out + 'net.pkl'
        with open(outputPkl, 'wb') as f:
            pickle.dump(net, f)
            
        # output the ground truth image names for later testing
        outputPkl = self.training_out + 'GT_imgnames.pkl'
        with open(outputPkl, 'wb') as f:
            pickle.dump(self.filteredNames, f)
        
        print('Network succesfully saved!')
        
        # if using cuda, clear and dump the memory usage
        if self.device =='cuda':
            del self.spike_rates
            gc.collect()
            torch.cuda.empty_cache()
    
    def validate(self): 
        
        # unpickle the network
        print('Unpickling the network')
        with open(self.training_out+'net.pkl', 'rb') as f:
             net = pickle.load(f)
        
        print('Validating network training')
        # set input spikes for training data from one location
        net['set_spks'][0] = ut.setSpikeRates(
                        self.imgs['training'][0:self.number_training_images],
                        self.ids['training'][0:self.number_training_images],
                        self.device,
                        [self.imWidth,self.imHeight],
                        True,
                        self.number_training_images,
                        self.number_modules,
                        self.intensity,
                        self.location_repeat)
        
        # correct place matches variable
        numcorrect = 0
        
        if self.validation:
            outconcat = []
        
        # run the test network on training data to evaluate network performance
        start = timeit.default_timer()
        for t in range(self.number_training_images):
            tonump = np.array([])
            bn.testSim(net,device=self.device)
            # output the index of highest amplitude spike
            tonump = np.append(tonump,
                               np.reshape(net['x'][-1].cpu().numpy(),
                              [1,1,int(self.number_training_images)]))
            nidx = np.argmax(tonump)
            
            if self.validation:
                outconcat.append(tonump.tolist())
            
            gt_ind = self.filteredNames.index(self.filteredNames[t])
            
            # adjust number of correct matches if GT matches peak output
            if gt_ind == nidx:
               numcorrect += 1
               
        end = timeit.default_timer()
        
        # network must match >95% of training places to be succesfull
        p100r = (numcorrect/self.number_training_images)*100
        testFlag = (p100r>75)
        if testFlag: # network training was successfull
            
            print('')
            print('Network training successful!')
            print('')
            
            print('Perfomance details:')
            print("-------------------------------------------")
            print('P@100R: '+str(p100r)+
                  '%  |  Query frequency: '+
                  str(round(self.number_training_images/(end-start),2))+'Hz')
            print('')
            
            # create output folder (if it does not already exist)
            if not os.path.isdir(self.training_out):
                os.mkdir(self.training_out)
            if not os.path.isdir(self.training_out+'images/'):
                os.mkdir(self.training_out+'images/')
            if not os.path.isdir(self.training_out+'images/training/'):
                os.mkdir(self.training_out+'images/training/')
            
            # plot the similarity matrix for network validation
            if self.validation:
                outconcat = np.array(outconcat)
                concatReshape = np.reshape(outconcat,
                                           (self.number_training_images,
                                            self.number_training_images))
            
                plot_name = "Similarity: network training validation"
                ut.plot_similarity(concatReshape,plot_name,
                                   self.training_out+'images/training/')
                
            # Reset network details
            net['sspk_idx'] = [0,0,0]
            net['step_num'] = 0
            net['spikes'] = [[],[],[]]
            net['x'] = [[],[],[]]
            net['set_spks'][0] = 0
            
           # plot the weight matrices
            cmap = plt.cm.magma
            
            print('Plotting weight matrices')
            # make weights folder
            weights_folder = self.output_folder+'/weights/'
            os.mkdir(weights_folder)
            
            # initial weights
            ut.plot_weights(self.init_weights[0],'Initial I->F Excitatory Weights',
                            cmap,self.number_modules,0.5,weights_folder)
            ut.plot_weights(self.init_weights[1], 'Initial I->F Inhibitory Weights',
                            cmap,self.number_modules,0.1,weights_folder)
            ut.plot_weights(self.init_weights[2], 'Initial F->O Excitatory Weights',
                            cmap,self.number_modules,0.01,weights_folder)
            ut.plot_weights(self.init_weights[3], 'Initial F->O Inhibitory Weights',
                            cmap,self.number_modules,0.01,weights_folder)
            
            # calculated weights
            ut.plot_weights(net['W'][0],'I->F Excitatory Weights',
                            cmap,self.number_modules,0.5,weights_folder)
            ut.plot_weights(net['W'][1], 'I->F Inhibitory Weights',
                            cmap,self.number_modules,0.1,weights_folder)
            ut.plot_weights(net['W'][2],'F->O Excitatory Weights',
                            cmap,self.number_modules,0.01,weights_folder)
            ut.plot_weights(net['W'][3], 'F->O Inhibitory Weights',
                            cmap,self.number_modules,0.01,weights_folder)
            
            print('Network formatting and saving...') 
            
            # Output the trained network
            outputPkl = self.training_out + 'net.pkl'
            with open(outputPkl, 'wb') as f:
                pickle.dump(net, f)
                
            # output the ground truth image names for later testing
            outputPkl = self.training_out + 'GT_imgnames.pkl'
            with open(outputPkl, 'wb') as f:
                pickle.dump(self.filteredNames, f)
            
            print('Network succesfully saved!')
        if self.device =='cuda':
            gc.collect()
            torch.cuda.empty_cache()
        '''
     Run the testing network
     '''

    def networktester(self):
         
        '''
        Network startup and initialization
        '''

        # unpickle the network
        print('Unpickling the network')
        with open(self.training_out+'net.pkl', 'rb') as f:
             net = pickle.load(f)
        
        print('Setting spike rates from loaded images')
        # calculate input spikes from training images
        
        net['set_spks'][0] = self.spike_rates['testing']
        
        # unpickle the ground truth image names
        with open(self.training_out+'GT_imgnames.pkl', 'rb') as f:
             GT_imgnames = pickle.load(f)
        
        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Running test network')    
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')
        
        # set number of correct places to 0
        numcorrect = 0
        net['spike_dims'] = self.input_layer
        numpconc = []
        start = timeit.default_timer()
        for t in range(self.number_testing_images):
            tonump = np.array([])
            bn.testSim(net,device=self.device)
            # output the index of highest amplitude spike
            tonump = np.append(tonump,np.reshape(net['x'][-1].cpu().numpy(),
                                       [1,1,int(self.number_training_images)]))
            gt_ind = GT_imgnames.index(self.filteredNames[t])
            nidx = np.argmax(tonump)

            if gt_ind == nidx:
               numcorrect += 1
            
            if self.validation: # get similarity matrix for PR curve generation
               numpconc.append(tonump.tolist())
        
        self.p100r = round((numcorrect/self.number_testing_images)*100,2)
        print('Number of correct matches P@100R - '+str(self.p100r)+'%')

        end = timeit.default_timer()
        queryHertz = self.number_testing_images/(end-start)
        print('System queried at '+str(round(queryHertz,2))+'Hz')
        
        # if self.validation = True, get PR information and plot similarity matrix
        if self.validation:
            
            numpconc = np.array(numpconc)
            
            # make the output folder
            folderName = self.output_folder+'/similarity/'
            os.mkdir(folderName)
            
            # reshape similarity matrix
            sim_mat = np.reshape(numpconc,(self.number_testing_images,
                                           self.number_training_images))
            
            # plot the similarity matrix
            fig = plt.figure()
            plt.matshow(sim_mat,fig,cmap='tab20c')
            plt.colorbar(label="Spike amplitude")
            fig.suptitle("Similarity matrix",fontsize = 12)
            plt.xlabel("Query",fontsize = 12)
            plt.ylabel("Databse",fontsize = 12)
            plt.savefig(folderName+'/sim_mat.pdf')
            plt.show()
            
            # generate the ground truth matrix
            GT = np.zeros((self.number_testing_images,self.number_training_images), dtype=int)
            for n in range(len(GT)):
                GT[n,n] = 1
            
            # plot the GT matrix
            fig = plt.figure()
            plt.matshow(GT,fig,cmap='tab20c')
            plt.colorbar(label="Spike amplitude")
            fig.suptitle("Ground Truth",fontsize = 12)
            plt.xlabel("Query",fontsize = 12)
            plt.ylabel("Databse",fontsize = 12)
            plt.savefig(folderName+'/GT.pdf')
            plt.show()    
            
            # get the P & R 
            P,R = createPR(sim_mat, GT, GT)
            
            # make the PR curve
            fig = plt.figure()
            plt.plot(R,P)
            fig.suptitle("Precision Recall curve",fontsize = 12)
            plt.xlabel("Recall",fontsize = 12)
            plt.ylabel("Precision",fontsize = 12)
            plt.show()
            
            # calculate the recall at N
            N_vals = [1,5,10,15,20,25]
            recallN = ut.recallAtN(sim_mat, GT, GT, N_vals)
            
            print('')
            for n, ndx in enumerate(recallN):
                print('Recall at N='+str(N_vals[n])+': '+str(round(ndx,2)))

            
        # if using cuda, clear and dump the memory usage
        if self.device =='cuda':
            gc.collect()
            torch.cuda.empty_cache()
    
    def sad(self):
        
        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Setting up Sum of Absolute Differences (SAD) calculations')    
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')
        
        sadcorrect = 0
        
        # load the training images
        self.location_repeat = 1 # switch to only do SAD on one dataset traversal
        self.fullTrainPaths = self.fullTrainPaths[1]
        self.test_true = False # testing images preloaded, load the training ones
        # load the training images
        self.imgs['training'], self.ids['training'] = ut.loadImages(self.test_true,
                                            self.fullTrainPaths,
                                            self.filteredNames,
                                            [self.imWidth,self.imHeight],
                                            self.num_patches,
                                            self.testPath,
                                            self.test_location)
        
        # port images to cpu
        #for n, ndx in enumerate(self.imgs['training']):
           # self.imgs['training'][n] = ndx.cpu()
        #for n, ndx in enumerate(self.imgs['testing']):
            #self.imgs['testing'][n] = ndx.cpu()
        
        # create database tensor
        for ndx, n in enumerate(self.imgs['training']):
            if ndx == 0:
                db = torch.unsqueeze(n,0)
            else:
                db = torch.concat((db,torch.unsqueeze(n,0)),0)
        
        def calc_sad(query, database, const):
            
            SAD = torch.sum(torch.abs((database * const) -  (query * const)),
                           (1,2),keepdim=True)
            for n in range(2):
                SAD = torch.squeeze(SAD,-1)
            return SAD
        
        # calculate SAD for each image to database and count correct number
        imgred = 1/(self.imWidth*self.imHeight)
        sad_concat = []
        print('Running SAD')
        correctidx = []
        incorrectidx = []
        start = timeit.default_timer()
        for n, q in enumerate(self.imgs['testing']):
            results = []
            pixels = torch.empty([])

            # create 3D tensor of query images
            for o in range(self.number_testing_images):
                if o == 0:
                    pixels = torch.unsqueeze(q,0)
                else:
                    pixels = torch.concat((pixels,torch.unsqueeze(q,0)),0)
                
            sad_score = calc_sad(pixels, db, imgred)
            
            best_match = np.argmin(sad_score.cpu().numpy())
            if n == best_match:
                sadcorrect += 1
                correctidx.append(n)
            else:
                incorrectidx.append(n)
            if self.validation:
                sad_concat.append(sad_score.cpu().numpy())
                
        end = timeit.default_timer()   
        p100r = round((sadcorrect/self.number_testing_images)*100,2)
        print('')
        print('Sum of absolute differences P@1: '+
              str(p100r)+'%')
        print('Sum of absolute differences queried at '
              +str(round(self.number_testing_images/(end-start),2))+'Hz')
        
        print('Network to sum of absolute differences ratio '+
              str(round(self.p100r/p100r,2)))
    
        if self.validation:
            # reshape similarity matrix
            sad_concat = np.array(sad_concat)
            sim_mat = np.reshape(sad_concat,(self.number_training_images,
                                               self.number_testing_images))
            
            # plot the similarity matrix
            fig = plt.figure()
            plt.matshow(sim_mat,fig,cmap='Greys')
            plt.colorbar(label="Sum of absolute differences")
            fig.suptitle("Similarity matrix - SAD",fontsize = 12)
            plt.xlabel("Query",fontsize = 12)
            plt.ylabel("Databse",fontsize = 12)
            plt.show()
            
            # generate the ground truth matrix
            GT = np.zeros((self.number_training_images,self.number_testing_images), dtype=int)
            for n in range(len(GT)):
                GT[n,n] = 1
                
            # get the P & R 
            sim_invert = 1/sim_mat # invert matrix so lowest SAD is highest value
            P,R = createPR(sim_invert, GT, GT)
            
            # make the PR curve
            fig = plt.figure()
            plt.plot(R,P)
            fig.suptitle("SAD Precision Recall curve",fontsize = 12)
            plt.xlabel("Recall",fontsize = 12)
            plt.ylabel("Precision",fontsize = 12)
            plt.show()
            
            # calculate the recall at N
            N_vals = [1,5,10,15,20,25]
            recallN = ut.recallAtN(sim_invert, GT, GT, N_vals)
            
            print('')
            for n, ndx in enumerate(recallN):
                print('Recall at N='+str(N_vals[n])+': '+str(round(ndx,2)))
'''
Run the network
'''        
if __name__ == "__main__":
    
    # Instantiate model
    model = snn_model()
    
    # check if the network has already been trained previously
    flg = model.checkTrainTest()
    
    # if user inputs 'y' to retrain network if network doesn't exist
    if flg == 'y':
        model.initialize('training') # Initializes the training network
        model.train() # Run network training (will check if already trained)
        model.validate() # Validates that the network trained properly
    
    # Tests the network
    model.initialize('testing') # Initializes the testing network
    model.networktester() # Test the network
    model.sad()


    #for n in range(1):
     #   model = snn_model()
      #  model.initialize('training')
       # model.train()
        #model.initialize('testing')
        #model.networktester()
        #model.sad()
        #gc.collect()
        #torch.cuda.empty_cache()