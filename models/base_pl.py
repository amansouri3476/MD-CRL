import pytorch_lightning as pl
from models.utils import update
import hydra

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

