import logging
#
import os.path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import torch
import json
matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
from dl_utils import *


from core.DownstreamEvaluator import DownstreamEvaluator
from optim.metrics.interp_metrics import *
import pandas as pd
from torchmetrics.functional import confusion_matrix, accuracy
from dl_utils.vizu_utils import AttributionLatentY, plot_conf_mat
from optim.metrics.auprc import AUPRC


ACDC_CLASSES = {'NOR':0,'MINF':1,'DCM':2,'HCM':3,'RV':4}

class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.encoder_model = self.model.encoder
        self.num_classes = self.model.num_classes
        self.dict_classes = self.model.dict_classes
        self.results_folder = self.model.results_folder


    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        self.model.load_state_dict(global_model)
        self.model.eval()

        dataloader = self.test_data_dict

        pred = []
        labels = []

        with torch.no_grad():
            for data, label, attr, full_attr in dataloader:
                x = data.to(self.device)
                y = label.to(self.device)

                y_hat = self.model(x)

                pred.append(y_hat.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())

        pred = np.concatenate(pred, 0)
        labels = np.concatenate(labels, 0)

        if self.num_classes == 2:
            labels = (labels>=1).astype(int)

        fig, df_acc = plot_conf_mat(pred, labels,
                                    self.dict_classes, False)

        tbl = wandb.Table(data=df_acc)
        wandb.log({"Test/Metrics": tbl})

        wandb.log({'Test/Confusion matrix': [wandb.Image(fig)]})

        index = 1
        labels_name = list(self.dict_classes.keys())
        attribution = AttributionLatentY(dataloader, labels_name, self.encoder_model,
                                         self.model,index, self.results_folder ,self.device)

        fig_global, fig_local = attribution.visualization()

        wandb.log({'Test' + f'/Attribution_global': [
            wandb.Image(fig_global, caption='Attribution_global')]})
       # wandb.log({'Test' + f'/Attribution_local_index_{index}': [
        #     wandb.Image(fig_local, caption= f'Attribution_local_index_{index}')]})

        attribution_score, _, _ = attribution.attribution()

        df_scores = pd.DataFrame()
        for i in range(self.num_classes):
            scores = attribution_score[i]
            columns_names = [f + f'_{i}' for f in attribution.feature_names]
            df = pd.DataFrame(scores[:,:len(columns_names)], columns= columns_names)

            df_scores = pd.concat([df_scores,df],axis=1)
        print(self.checkpoint_path)

        run = self.checkpoint_path.split('/')[-1]
        output_path = self.results_folder + f'/shap/values_{run}.csv'
        print(output_path)
        df_scores.to_csv(output_path, index=False)

