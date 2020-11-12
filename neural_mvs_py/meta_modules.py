'''Modules for hypernetwork experiments, Paper Sec. 4.4
'''

import torch
from torch import nn
from collections import OrderedDict
import modules
import math

import numpy as np

from neural_mvs.modules import *


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        param_idx=0
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            print("params name ", name, " params size si ", param.size())

            hn = modules.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                                 outermost_linear=True, nonlinearity='relu')
                                #  outermost_linear=True, nonlinearity='selu')
                                #  outermost_linear=True, nonlinearity='elu')
            self.nets.append(hn)

            if 'weight' in name:
                if param_idx==0:
                    self.nets[-1].net[-1].apply(lambda m: hyper_weight_init_first_siren_layer(m, param.size()[-1]))
                else:
                    self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

            param_idx+=1

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        # print("computing hyperparams")
        i=0
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            # print("param_shape si ", param_shape, " batch_param_shape is ", batch_param_shape)
            # params[name] = net(z).reshape(batch_param_shape)
            params[name] = net(z).reshape(param_shape)
            # print("param has mean and std ", params[name].mean(), params[name].std() )

            # print("params for first is ",params[name] )

            i+=1
        return params


#following the initialization scheme by
class HyperNetworkPrincipledInitialization(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        param_idx=0
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            print("params name ", name, " params size si ", param.size())

            # hn = modules.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
            #                      num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
            #                      outermost_linear=True, nonlinearity='relu')
            hn = torch.nn.Sequential(
                BlockLinear(in_channels=hyper_in_features, out_channels=512, bias=True, activ=torch.relu ),
                BlockLinear(in_channels=512, out_channels=512, bias=True, activ=torch.relu ),
                BlockLinear(in_channels=512, out_channels=512, bias=True, activ=torch.relu ),
                BlockLinear(in_channels=512, out_channels=512, bias=True, activ=torch.relu ),
                torch.nn.Linear(in_features=512, out_features=int(torch.prod(torch.tensor(param.size()))), bias=True)
            )

            self.nets.append(hn)

            if 'weight' in name:
                in_features_main_net=param.shape[1]
                self.nets[-1][-1].apply(lambda m: principled_init_for_predicting_weights(m, in_features_main_net ))
            elif 'bias' in name:
                in_features_main_net=param.shape[0]
                # self.nets[-1][-1].apply(lambda m: principled_init_for_predicting_weights(m, in_features_main_net ))
                self.nets[-1].apply(lambda m: principled_init_for_predicting_bias(m))

            param_idx+=1

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        # print("computing hyperparams")
        i=0
        # print("--------------------------------------")
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            # batch_param_shape = (-1,) + param_shape
            # print("param_shape si ", param_shape, " batch_param_shape is ", batch_param_shape)
            # params[name] = net(z).reshape(batch_param_shape)
            # print("z shape is ", z.shape)


            # z_scaled=z*6
            # z_scaled=z*1000
            # z_scaled=z*13
            # z_scaled=z*19
            z_scaled=z*2.5
            # print("HYPERNET: z has mean ", z_scaled.mean().item(), " var", z_scaled.var().item(),"Std ", z_scaled.std().item() )
            weight= net(z_scaled).reshape(param_shape)
            # if "net" in name and "weight" in name:
                # std=weight.std()
                # print("std is ", std)
                # weight=weight/std*0.115
            params[name] = weight
            # print("param has mean and std ", params[name].mean(), params[name].std() )

            # print("params for first is ",params[name] )

            # if "weight" in name: 
                    # print(name,"params have mean and std ", params[name].mean(), " std ", params[name].std() )

            i+=1
        return params


class HyperNetworkIncremental(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        self.predictor_or_siren_weights = None
        self.siren_channels=None
        param_idx=0
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())


            print("hypo params name ", name, " params size si ", param.size())

            if "net" in name and "weight" in name and self.predictor_or_siren_weights==None: #we are predicting the weight or bias for the siren
                self.siren_channels=param.shape[0] #the output of the first parameter of the siren will be some thing like 128 or 256
                print("weight of siren with output channels ", self.siren_channels)
                self.predictor_or_siren_weights = modules.PredictorSirenIncremental(  nonlinearity='relu'  )

            hn = modules.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                                 outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            # if 'weight' in name:
                # if param_idx==0:
                    # print("init to first siren layer")
                    # self.nets[-1].net[-1].apply(lambda m: hyper_weight_init_first_siren_layer(m, param.size()[-1]))
                    # self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
                # else:
                    # self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            # elif 'bias' in name:
                # self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

            param_idx+=1

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        # print("computing hyperparams")
        i=0
        nr_times_called_predictor_of_siren=0
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            print("predicting for ", name, "with params shape", param_shape)
            
            if "net" in name: #we are predicting the weight or bias for the siren
                if "weight" in name:
                    print("predictiing with an incremental siren")
                    #we predict here both the weight and the bias
                    # print("predicting for ", name, "with params shape", param_shape)
                    is_first=nr_times_called_predictor_of_siren==0
                    if is_first:
                        prev_weights= torch.zeros([ self.siren_channels, self.siren_channels, 1, 1 ], dtype=torch.float32, device=torch.device("cuda"))
                        prev_bias= torch.zeros([  self.siren_channels ], dtype=torch.float32, device=torch.device("cuda"))
                    else:
                        prev_weights= params[ "net."+str(nr_times_called_predictor_of_siren-1)+".0.conv.0.weight" ]
                        prev_bias= params[ "net."+str(nr_times_called_predictor_of_siren-1)+".0.conv.0.bias" ]
                    print("previous siren weights ", prev_weights.shape)
                    print("previous siren bias ", prev_bias.shape)
                    new_weight, new_bias =self.predictor_or_siren_weights(z, prev_weights, prev_bias, param_shape) 

                    #set the new weight and bias 
                    params["net."+str(nr_times_called_predictor_of_siren)+".0.conv.0.weight"]= new_weight
                    params["net."+str(nr_times_called_predictor_of_siren)+".0.conv.0.bias"]= new_bias

                    #double check everything
                    # print("whattttttttttttttttttttttt", new_weight.shape, " param shape si ", param_shape)
                    if new_weight.shape!=param_shape:
                        print(" ERROR new weight has shape", new_weight.shape, " but it should have shape ", param_shape)
                        exit(1)

                    nr_times_called_predictor_of_siren+=1
                if "bias" in name:
                    #DO NOTHING because we already predicted the bias while we predicted the weights
                    pass
            else:
                print("predictiing with a normal fcblock")
                params[name] = net(z).reshape(param_shape)

            i+=1
        print("returning params")
        return params


class NeuralProcessImplicit2DHypernet(nn.Module):
    '''A canonical 2D representation hypernetwork mapping 2D coords to out_features.'''
    def __init__(self, in_features, out_features, image_resolution=None, encoder_nl='sine'):
        super().__init__()

        latent_dim = 256
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=2)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        self.set_encoder = modules.SetEncoder(in_features=in_features, out_features=latent_dim, num_hidden_layers=2,
                                              hidden_features=latent_dim, nonlinearity=encoder_nl)
        print(self)

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def get_hypo_net_weights(self, model_input):
        pixels, coords = model_input['img_sub'], model_input['coords_sub']
        ctxt_mask = model_input.get('ctxt_mask', None)
        embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            pixels, coords = model_input['img_sub'], model_input['coords_sub']
            ctxt_mask = model_input.get('ctxt_mask', None)
            embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)
        return {'model_in':model_output['model_in'], 'model_out':model_output['model_out'], 'latent_vec':embedding,
                'hypo_params':hypo_params}


class ConvolutionalNeuralProcessImplicit2DHypernet(nn.Module):
    def __init__(self, in_features, out_features, image_resolution=None, partial_conv=False):
        super().__init__()
        latent_dim = 256

        if partial_conv:
            self.encoder = modules.PartialConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        else:
            self.encoder = modules.ConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=2)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False


############################
# Initialization schemes
def hyper_weight_init_first_siren_layer(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # m.weight.data = m.weight.data / 1.e2
        m.weight.data = m.weight.data / 1

        # print("weight is",  m.weight.data )

        with torch.no_grad():
            # m.weight.uniform_(-np.sqrt(6 / in_features_main_net)/200 , np.sqrt(6 / in_features_main_net)/200 )
            m.weight.uniform_(-1 / in_features_main_net /4, 1 / in_features_main_net /4 )

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)
            m.bias.data = m.bias.data / 1



def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # m.weight.data = m.weight.data / 1.e2
        m.weight.data = m.weight.data / 200

        with torch.no_grad():
            m.weight.uniform_(-np.sqrt(6 / in_features_main_net)/150 , np.sqrt(6 / in_features_main_net)/150 )

        # print("weight is",  m.weight.data )

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)
            m.bias.data = m.bias.data / 200


def principled_init_for_predicting_weights(m, in_features_main_net):
    if hasattr(m, 'weight'):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        # var= 1.0/(fan_in * in_features_main_net) *2 #the x2 is because we want the variance to increase from the 0.5 which we have after a sine activation to a 1.0 
        var= 1.0/(fan_in * in_features_main_net)
        print("fan in is ", fan_in, " in_features_main_net ", in_features_main_net)
        print("initializing weight with var ", var)
        std= np.sqrt(var)
        bound = math.sqrt(3.0) * std 
        with torch.no_grad():
            m.weight.uniform_(-bound, bound)

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.data.fill_(0.0)

def principled_init_for_predicting_bias(m):
    if hasattr(m, 'weight'):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        print("initialize for bias prediction fan in is ", fan_in, )
        # var= 1.0/(fan_in)
        var= 1.0/(fan_in)
        std= np.sqrt(var)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            m.weight.uniform_(-bound, bound)

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.data.fill_(0.0)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # m.weight.data = m.weight.data / 1.e2
        m.weight.data = m.weight.data 

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)
            m.bias.data = m.bias.data 
