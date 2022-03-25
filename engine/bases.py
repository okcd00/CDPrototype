import logging
import pytorch_lightning as pl

from solver.build import make_optimizer, build_lr_scheduler


class BaseTrainingEngine(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self._logger = logging.getLogger(cfg.MODEL.NAME)

    def configure_optimizers(self):
        optimizer = make_optimizer(self.cfg, self)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [scheduler]
    
    def on_validation_epoch_start(self) -> None:
        # only log once on multi-gpus
        if self.trainer.is_global_zero:
            self._logger.info('\n=====Valid=====\n')

    def on_test_epoch_start(self) -> None:
        # only log once on multi-gpus
        if self.trainer.is_global_zero:
            self._logger.info('\n=====Testing=====\n')
