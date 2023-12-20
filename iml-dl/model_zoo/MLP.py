import glob

import pytorch_lightning as pl
import torch

import yaml

from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import confusion_matrix, accuracy

from dl_utils.config_utils import check_config_file, import_module


class MLP(pl.LightningModule):

    def __init__(
        self,
        folder_name: str,
        data_dir: str,
        latent_dim: int,
        dict_classes: dict,
        blocked_latent_features: int,
        fix_weights=True,
        hidden_neurons = 512,
        n_layers = 3,
        relu = True,
        dropout =0
    ):

        super().__init__()

        self.dict_classes = dict_classes
        self.num_classes = len(self.dict_classes)# if len(self.dict_classes)>2 else 1
        self.results_folder = data_dir + folder_name

        self.hidden_neurons = hidden_neurons
        self.n_layers = n_layers
        self.relu = relu
        self.dropout = dropout

        path_pt = data_dir + folder_name + "/best_model.pt"
        weights = torch.load(path_pt)

        #config_folder = glob.glob(f'./wandb/*{folder_name}/files/')
        model_config = data_dir + folder_name + "/config.yaml"
        stream_file = open(model_config, 'r')
        config = yaml.load(stream_file, Loader=yaml.FullLoader)
        self.dl_config = check_config_file(config)

        model_class = import_module(self.dl_config['model']['module_name'], self.dl_config['model']['class_name'])
        self.encoder = model_class(**(self.dl_config['model']['params']))

        self.model_name = self.dl_config['model']['module_name'].split('.')[1]

        if 'latent_dim' in self.dl_config['model']['params'].keys():
            self.latent_dim = self.dl_config['model']['params']['latent_dim'] # parameters of the encoder
        elif 'z_dim' in self.dl_config['model']['params'].keys():
            self.latent_dim = self.dl_config['model']['params']['z_dim'] # parameters of the encoder
        elif 'zdim' in self.dl_config['model']['params'].keys():
            self.latent_dim = self.dl_config['model']['params']['zdim']
        else:
            self.latent_dim = latent_dim # parameters in head config

        self.blocked_latent_features = list(range(blocked_latent_features,self.latent_dim))

        self.kept_latent_features = torch.tensor(
            [x for x in list(range(0, self.latent_dim)) if x not in self.blocked_latent_features]
        )

        self.encoder.load_state_dict(weights['model_weights'])

        #layer_enc_names = [name.split('.')[0] for name, param in self.model.named_parameters() if 'enc' in name]
        if fix_weights:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder.eval()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(len(self.kept_latent_features), self.hidden_neurons, bias =True))
        if self.relu:
            self.layers.append(nn.ReLU())
        if self.dropout > 0:
            self.layers.append(nn.Dropout(self.dropout))

        # Hidden layers
        for _ in range(self.n_layers - 1):
            self.layers.append(nn.Linear(self.hidden_neurons, self.hidden_neurons, bias = True))
            if self.relu:
                self.layers.append(nn.ReLU())
            if self.dropout > 0:
                self.layers.append(nn.Dropout(self.dropout))

        # Output layer
        self.layers.append(nn.Linear(self.hidden_neurons, self.num_classes, bias = True))

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)


        """
        self.fc_first = nn.Linear(len(self.kept_latent_features), self.hidden_neurons, bias=True)
        self.fc1 = nn.Linear(self.hidden_neurons, self.hidden_neurons, bias=True)
        self.fc2 = nn.Linear(self.hidden_neurons, self.hidden_neurons, bias=True)
        self.fc3 = nn.Linear(self.hidden_neurons, self.hidden_neurons, bias=True)
        self.fc_last = nn.Linear(self.hidden_neurons, self.num_classes, bias=True)

        self.dropout = nn.Dropout(0.5)

        torch.nn.init.kaiming_uniform_ (self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)
        """
    def forward(self, x):
        """Predicts from encoded or not encoded image.

        Parameters
        ----------
        x : torch.Tensor
            Image or latent representation.

        Returns
        -------
        torch.Tensor
            Prediction.
        """


        if len(x.shape) >=3:
            if self.model_name == 'beta_vae_higgings':
                _, f_results = self.encoder.encode(x)
                x = f_results['z']

            elif self.model_name == 'soft_intro_vae_daniel':
                #len(x.shape) >= 3:

                mu, logvar = self.encoder.encode(x)
                """
                z_dist = self.encoder.encode(x)
                if len(z_dist.shape) == 4:
                    x = torch.squeeze(z_dist)
                elif len(z_dist)>1:
                    x,_,_,_ = self.encoder.reparameterize(z_dist[0],z_dist[1],z_dist[2])
                else:
                    x, _, _ = self.encoder.reparametrize(z_dist)
                """
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                x =  mu + eps * std
            else:
                pass
        else:
            pass

        if len(self.blocked_latent_features) > 0:
            x = x.index_select(1, self.kept_latent_features.to(x.device))

        for layer in self.layers:
            x = layer(x)
        y_hat = F.softmax(x, dim=1)

        return y_hat



"""
 x = F.relu(self.fc_first(x))
#x = self.dropout(x)
x = F.relu(self.fc1(x))
#x = self.dropout(x)
#x = F.relu(self.fc2(x))
#x = self.dropout(x)
#x = F.relu(self.fc3(x))
#x = self.dropout(x)
"""