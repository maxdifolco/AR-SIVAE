import logging
import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb

from torch.nn import L1Loss, MSELoss
#
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import PeakSignalNoiseRatio

#
import lpips
from core.DownstreamEvaluator import DownstreamEvaluator
from optim.metrics.interp_metrics import *
from dl_utils.vizu_utils import *


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.attributes_dict = test_data_dict.dataset.dataset.attributes_dict
        self.attributes_idx = test_data_dict.dataset.dataset.attributes_idx
        self.name = name

        self.criterion_MSE = MSELoss().to(self.device)
        self.save_MSE = MSELoss(reduce=False).to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=False, lpips=True).to(self.device)

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        self.model.load_state_dict(global_model)
        self.model.eval()

        latent_codes, full_attributes, predictions, labels, rec_error = self.compute_latent_representations()
        rl_metrics = compute_rl_metrics(self.checkpoint_path, latent_codes.detach().cpu().numpy(), full_attributes, self.attributes_idx)

        #  Interpretability metrics
        df = pd.DataFrame(rl_metrics['interpretability'])
        tbl = wandb.Table(data=df)
        wandb.log({f"{self.name}/Interpretability metrics": tbl})

        df_metrics = pd.DataFrame()
        for key in rl_metrics.keys():
            if key != 'interpretability':
                df_metrics[key] = [rl_metrics[key]]

        for k in rec_error.keys():
            df_metrics[k] = np.mean(rec_error[k])
        #df_metrics['MSE'] = np.mean(rec_error['MSE'])
        #df_metrics['SSIM'] = np.mean(rec_error['SSIM'])
        #df_metrics['PSNR'] = np.mean(rec_error['PSNR'])

        tbl = wandb.Table(data=df_metrics)
        wandb.log({f"{self.name}/Metrics": tbl})

        dim_list = [rl_metrics['interpretability'][K][0] for K in rl_metrics['interpretability'].keys() if 'mean' not in K]

        save_df = pd.DataFrame(rec_error['MSE'], columns=['MSE'])
        save_df['SSIM'] = rec_error['SSIM']
        save_df['PSNR'] = rec_error['PSNR']
        save_df['LPIPS'] = rec_error['LPIPS']
      
        print(self.checkpoint_path + f'/{self.name}_error.csv')
        save_df.to_csv(self.checkpoint_path + f'/{self.name}_error.csv')

        self.show_latent_space(latent_codes, dim_list = dim_list)
   

    def show_latent_space(self, latent_codes, dim_list = [0,1,2], dim_plot_2d = [0,1]):

        fig = plot_latent_reconstructions(self.model, self.test_data_dict, self.device, num_points=8)
        wandb.log({f'{self.name}/Reconstruction examples': [
            wandb.Image(fig, caption=f'Test_reconstruction')]})

        range_value = 5.0
        fig = plot_latent_interpolations(self.model,latent_codes[:1,:], dim_list=dim_list,
                                         num_points=4, range_value=range_value)
        wandb.log({f'{self.name}/_Latent dimensions': [
             wandb.Image(fig, caption= f'{range_value}' + 'Latent_dim_' + '_'.join(str(dim) for dim in dim_list))]})

        dim1, dim2 = dim_plot_2d
        fig = plot_latent_interpolations2d(self.model,latent_codes[:1,:], dim1=dim1, dim2=dim2, num_points=5)
        wandb.log({f'{self.name}/_Dimensions_' + str(dim1) + '_' + str(dim2): [
            wandb.Image(fig, caption=f'Latent_dim_' + str(dim1) + '_' + str(dim2))]})

    def compute_latent_representations(self):
        

        dataset = self.test_data_dict

        idx = 0
        #z_dim = self.model.z_dim

        latent_codes = []
        labels, predictions = [], []
        attributes, full_attributes = [], []
        mse_loss, ssim_, psnr_, lpips_ = [], [], [], []

        PSNR = PeakSignalNoiseRatio().to(self.device)

        with torch.no_grad():
            for data, label, attr, full_attr in dataset:
                nr_slices, c, width, height, = data.shape
                x = data.view(nr_slices, c, width, height)

                x = x.to(self.device)
                rec, f_result = self.model(x)

                MSE = self.criterion_MSE(rec,x)
                save_MSE = self.save_MSE(rec,x)

                tmp = save_MSE.detach().cpu().numpy()
                save_MSE = [np.mean(tmp[i,:,:,:]) for i in range(tmp.shape[0])]
                mse_loss.append(save_MSE)

                for i in range(np.shape(data)[0]):
                    x_i = x[i].cpu().detach().numpy()
                    x_rec_i = rec[i].cpu().detach().numpy()
                    S = ssim(x_rec_i, x_i, channel_axis=0 ,data_range=1.0)
                    ssim_.append(S)
                    psnr_.append(PSNR(rec[i], x[i]).cpu().detach().numpy())
                    lpips_value = torch.squeeze(self.l_pips_sq(torch.squeeze(rec[i]), torch.squeeze(x[i]), normalize=True, retPerLayer=False))
                    lpips_.append(lpips_value.cpu().detach().numpy().tolist())

                if len(f_result['z'].size()) > 2:
                    latent_codes.append(torch.squeeze(f_result['z']))
                else:
                    latent_codes.append(f_result['z'])
                attributes.append(attr.detach().cpu().numpy())
                full_attributes.append(full_attr.detach().cpu().numpy())
                labels.append(label)

        latent_codes = torch.cat(latent_codes,0)
        attributes = np.concatenate(attributes, 0)
        full_attributes = np.concatenate(full_attributes,0)
        labels = torch.cat(labels,0)
        rec_error = {'MSE': np.concatenate(mse_loss,0), 'SSIM': ssim_, 'PSNR': psnr_, 'LPIPS': lpips_}

        #rec_error = np.mean(mse_loss)
        z_ = latent_codes.detach().cpu().numpy()

        return latent_codes, full_attributes, predictions, labels, rec_error
