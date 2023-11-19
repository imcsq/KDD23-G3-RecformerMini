import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

class LitWrapper(pl.LightningModule):
    """
    A PyTorch Lightning wrapper for Recformer.

    by overwritten the training/validation steps in Lightning; OR implement the API to train/validate/optimizer
    """
    def __init__(self, 
                model: nn.Module,
                learning_rate: float = 5e-5,
                warmup_steps: int = 0,
                weight_decay: float = 0.0
                ):
        super().__init__()
        
        self.hparams.learning_rate = learning_rate
        self.hparams.warmup_steps = warmup_steps
        self.hparams.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # this is over-written ?重写parent class. 
        # otherwise where is this function called?

        outputs = self(**batch)

        loss = outputs.loss
        correct_num = outputs.cl_correct_num    # customed attribute, to check
        total_num = outputs.cl_total_num        # customed attribute, to check

        accuracy = 0.0
        if total_num > 0:
            accuracy = correct_num / total_num
        # [?] implement theligh metric of accuracy? 

        self.log_dict({'train_loss': loss, 'accuracy': accuracy, 'mlm_loss': outputs.misc['mlm_loss']}, on_step=True, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        correct_num = outputs.cl_correct_num
        total_num = outputs.cl_total_num

        accuracy = 0.0
        if total_num > 0:
            accuracy = correct_num / total_num

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_mlm_loss': outputs.misc['mlm_loss']}, on_epoch=True, prog_bar=True)
        # print("val res at", self.current_epoch, self.global_step, {'val_loss': loss, 'accuracy': accuracy, 'mlm_loss': outputs.misc['mlm_loss']})
        self.validation_step_outputs.append({'val_loss': loss, 'val_accuracy': accuracy, 'val_mlm_loss': outputs.misc['mlm_loss']})

    def on_validation_epoch_end(self):  # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.
        # [?] where is the item-item contrastive learning loss?
        # [to check] is the item-item contrastive learning loss implemented in the architecture part?
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_val_mlm_loss = torch.stack([x['val_mlm_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_accuracy': avg_val_accuracy, 'avg_val_mlm_loss': avg_val_mlm_loss}
        print("val res at", self.current_epoch, self.global_step, tensorboard_logs)
        self.log_dict(tensorboard_logs)
        self.validation_step_outputs.clear()

        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        # optimizer the weights with AdamW and defined learning rate
        # what is the functin of optimizer and scheduler?
        """
        optimizer: method to change the weights / learning rates
            Optimizers guide the training process towards convergence, minimizing (or in some cases, maximizing) a loss function.
            The optimizer takes the gradients (obtained from backpropagation) and performs weight updates using these gradients. Different optimizers use different strategies for weight updates.
        scheduler:
            adjust the learning rate during training

        We have set the learning rate as hyperparameter. What is the relation between learning rate and scheduler when taking about the learning rate?
        """

        # what is the advantage of using AdamW rather than use SGD?
        """
            AdamW with adaptive learning rate, more generalizability, and less sensitive to the hyperparameter choice

        SGD is simpler.
        """
        # what is weight decay? is it to udpate weights?
        """
            Weight decay is a regularization technique used in the training of neural networks. It's not directly about updating the weights in the way that backpropagation and optimization do, but rather it's a method to prevent overfitting by penalizing large weights. 
        """
        # relation between scheduler and our setting hyper-parameter learning rate?
        """
        Interaction Between Learning Rate and Scheduler:
            * The initial learning rate set as a hyperparameter is the $starting point$ for the training. 
                The scheduler then adjusts this rate according to its scheduling strategy.
            * 
            Benefits of Using a Scheduler:
                - Faster Convergence: Initially high learning rates can lead to faster convergence.
                - Avoiding Local Minima: A varying learning rate can help in escaping local minima.
                - Fine-Tuning: Lower learning rates in later phases allow for more precise adjustments to the model’s weights.
        """
        
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]