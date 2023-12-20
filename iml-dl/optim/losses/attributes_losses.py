import torch

def compute_reg_loss(z, attr,factor):
    reg_loss = 0.0
    reg_dim_real = attr.size()[1]
    for dim in range(reg_dim_real):
        x_ = z[:, dim]
        reg_loss += reg_loss_sign(x_, attr[:, dim], factor)

    return reg_loss

def reg_loss_sign(latent_code, attribute, factor):
    """
    Computes the regularization loss given the latent code and attribute
    Args:
        latent_code: torch Variable, (N,)
        attribute: torch Variable, (N,)
        factor: parameter for scaling the loss
    Returns
        scalar, loss
    """
    # compute latent distance matrix
    latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
    lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

    # compute attribute distance matrix
    attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
    attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

    # compute regularization loss
    loss_fn = torch.nn.L1Loss()
    lc_tanh = torch.tanh(lc_dist_mat * factor).cpu()
    attribute_sign = torch.sign(attribute_dist_mat)
    sign_loss = loss_fn(lc_tanh, attribute_sign.float())

    return sign_loss

class AttriVAE_loss:
    def __init__(self, beta, gamma, factor, alpha_mlp=1.0):
        super(AttriVAE_loss,self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.factor = factor
        self.alpha = alpha_mlp

    def __call__(self, x_recon, x, f_results, labels, attr):

        recon_loss = reconstruction_loss(x_recon, x, 1.0, dist='gaussian')

        log_var = f_results['z_logvar']
        mu = f_results['z_mu']
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        reg_loss = regularization_loss(f_results['z'], attr, x_recon.size()[0], self.gamma, self.factor)
        loss = recon_loss + self.beta * kld_loss + reg_loss

        #print(f'Recon_loss: {recon_loss}; mlp_loss: {mlp_loss/self.alpha}'
         #     f', kl_loss: {kl_loss/self.beta}, reg_loss: {reg_loss/self.gamma}')
        return loss
