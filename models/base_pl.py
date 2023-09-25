import pytorch_lightning as pl
from models.utils import update
import torch
import hydra
from omegaconf import OmegaConf
from torch.nn import functional as F
from models.utils import penalty_loss_minmax, penalty_loss_stddev, hinge_loss, penalty_domain_classification

class BasePl(pl.LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Setup for all computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore=["datamodule", "additional_logger"], logger=False) # This is CRUCIAL, o.w. checkpoints try to pickle 
        # datamodule which not only takes a lot of space, but raises error because in contains generator
        # objects that cannot be pickled.
        if kwargs.get("hparams_overrides", None) is not None:
            # Overriding the hyper-parameters of a checkpoint at an arbitrary depth using a dict structure
            hparams_overrides = self.hparams.pop("hparams_overrides")
            update(self.hparams, hparams_overrides)

        self.num_domains = self.hparams.get("num_domains", 4)
        self.z_dim_invariant_model = self.hparams.get("z_dim_invariant", 4)

        self.penalty_criterion = self.hparams.get("penalty_criterion", {"minmax": 0., "stddev": 0., "domain_classification": 0.})
        if self.penalty_criterion and self.penalty_criterion["minmax"]:
            # self.penalty_loss = penalty_loss_minmax
            self.loss_transform = self.hparams.get("loss_transform", "mse")
        # elif self.penalty_criterion:
        #     # self.penalty_loss = penalty_loss_stddev
        if self.penalty_criterion and self.penalty_criterion["domain_classification"]:
            # self.penalty_loss = penalty_domain_classification
            from models.modules.multinomial_logreg import LogisticRegressionModel
            from torch import nn
            self.multinomial_logistic_regression = LogisticRegressionModel(self.z_dim_invariant_model, self.num_domains)
            self.multinomial_logistic_regression = self.multinomial_logistic_regression.to(self.device)
            self.domain_classification_loss = nn.CrossEntropyLoss()
        # else:
        #     raise ValueError(f"penalty_criterion {self.penalty_criterion} not supported")
        self.top_k = self.hparams.get("top_k", 5)
        self.stddev_threshold = self.hparams.get("stddev_threshold", 0.1)
        self.stddev_eps = self.hparams.get("stddev_eps", 1e-4)
        self.hinge_loss_weight = self.hparams.get("hinge_loss_weight", 0.0)

    def loss(self, x, x_hat, z_hat, domains):

        # x, x_hat: [batch_size, x_dim]
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
        penalty_loss_value = torch.tensor(0., device=self.device)
        hinge_loss_value = hinge_loss(z_hat, domains, self.num_domains, self.z_dim_invariant_model, self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight) if self.hinge_loss_weight > 0. else torch.tensor(0., device=x.device)

        if self.penalty_criterion and self.penalty_criterion["minmax"]:
            penalty_loss_args = [self.top_k, self.loss_transform]
            penalty_loss_value_ = penalty_loss_minmax(z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        if self.penalty_criterion and self.penalty_criterion["stddev"]:
            penalty_loss_args = []
            penalty_loss_value_ = penalty_loss_stddev(z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        if self.penalty_criterion and self.penalty_criterion["domain_classification"]:
            penalty_loss_args = [self.multinomial_logistic_regression, self.domain_classification_loss]
            penalty_loss_value_ = penalty_domain_classification(z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        
        penalty_loss_value = penalty_loss_value * self.penalty_weight
        hinge_loss_value = hinge_loss_value * self.hinge_loss_weight
        loss = reconstruction_loss + penalty_loss_value + hinge_loss_value
        return loss, reconstruction_loss, penalty_loss_value, hinge_loss_value
    
    def configure_optimizers(self):

        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                   , params
                                                                  )
        
        if self.hparams.get("scheduler_config"):
            # for pytorch scheduler objects, we should use utils.instantiate()
            if self.hparams.scheduler_config.scheduler['_target_'].startswith("torch.optim"):
                scheduler = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer)

            # for transformer function calls, we should use utils.call()
            elif self.hparams.scheduler_config.scheduler['_target_'].startswith("transformers"):
                scheduler = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer)
            
            else:
                raise ValueError("The scheduler specified by scheduler._target_ is not implemented.")
                
            scheduler_dict = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
                                                    , resolve=True)
            scheduler_dict["scheduler"] = scheduler

            return [optimizer], [scheduler_dict]
        else:
            # no scheduling
            return [optimizer]

