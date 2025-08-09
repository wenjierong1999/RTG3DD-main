import torch
import time
from src.dist_utils import utils
from pytorch_lightning.utilities import rank_zero_only
from torch.cuda.amp import autocast as autocast


class MyBaseTrainer:

    def __init__(
        self,
        min_epochs,
        max_epochs,
        distributed=True,
        logger=None, # see src/logger/MyBaseLogger
        device='cuda', 
        step=0, # global step counter
        ckpt_resume=False, # If true, resume training from a ckpt
        modelmodule=None,
    ):
        super().__init__()
        self.distributed = distributed
        self.device = device
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.logger = logger
        self.step = step
        self.ckpt_resume = ckpt_resume
        self.use_amp = False
        if self.ckpt_resume:
            self.min_epochs = self.load_model(modelmodule)


    def train_one_epoch(self, data_loader, modelmodule):
        """
        Train the model for one epoch over the data_loader.
        """
        for batch in data_loader:
            torch.cuda.synchronize()
            time_st_batch = time.time()  # Start timing the iteration

            # Zero gradients before backpropagation
            modelmodule.net.zero_grad()
            modelmodule.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Forward and loss computation, step-wise
                batch_stat, batch_media = modelmodule.training_step(batch, self.step) # training_step -> return:{"loss": loss}, {"image": colors}
                loss = batch_stat['loss']
                image = batch_media['image']
            # Log input/output images
            self.logger.save_image(image, self.step)

            # Optionally log render images (e.g., output of rendering networks)
            if 'render_image' in batch_media.keys():
                render_image = batch_media['render_image']
                self.logger.save_image(render_image, self.step, name='render')

            # Backpropagation and optimizer step
            loss.backward()
            modelmodule.optimizer.step()
            modelmodule.optimizer.zero_grad()
            # Update EMA (Exponential Moving Average) model if applicable
            if modelmodule.net_ema is not None:
                modelmodule.net_ema.update(modelmodule.net)

            # Increase global step
            self.step += 1

            # Timing for the iteration
            torch.cuda.synchronize()
            time_iter = time.time() - time_st_batch
            batch_stat['time_iter'] = time_iter

            # Log statistics (loss, time, etc.)
            self.logger.print_stat(batch_stat, self.step)
            
            # release unallocated cuda memory
            torch.cuda.empty_cache()
    
    def train(self, datamodule, modelmodule):
        """
        Main training loop over multiple epochs.
        """
        start_epoch = self.min_epochs
        end_epoch = self.max_epochs

        # Set model to training mode
        modelmodule.net.train()

        # Get data loaders
        data_loader_train = datamodule.train_dataloader()
        data_loader_val = datamodule.val_dataloader()  # Not used here but can be used for validation

        print(f"Start training for {self.max_epochs} epochs")

        epoch_times = []

        for epoch in range(start_epoch, end_epoch + 1):

            epoch_start_time = time.time()

            if self.distributed:
                # For distributed training, set the epoch to shuffle data differently across workers
                data_loader_train.sampler.set_epoch(epoch)

            # Train for one epoch
            self.train_one_epoch(data_loader_train, modelmodule)

            # Save checkpoint
            self.save_model(epoch, modelmodule)

            epoch_end_time = time.time()
            # Compute time taken for this epoch
            time_elapsed = epoch_end_time - epoch_start_time
            epoch_times.append(time_elapsed)
            avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
            remaining_epochs = end_epoch - (epoch)
            est_remaining_time = avg_time_per_epoch * remaining_epochs
            print(f"Epoch {epoch}/{self.max_epochs} finished in {time_elapsed:.2f} seconds. Estimated remaining time: {est_remaining_time / 60:.2f} minutes.")
    
    def load_model(self, modelmodule):
        '''
        Resume training by loading model and optimizer state from checkpoint.
        Returns the epoch number to resume from.
        '''
        net = modelmodule.net
        net_without_ddp = modelmodule.net_without_ddp  # Non-distributed version of the model
        net_ema = modelmodule.net_ema  # EMA model
        optimizer = modelmodule.optimizer
        output_dir = self.ckpt_resume  # Path to the checkpoint directory

        # Load model, optimizer, and EMA state
        epoch, step = utils.auto_load_model(
            output_dir=output_dir,
            model=net,
            model_without_ddp=net_without_ddp,
            optimizer=optimizer,
            model_ema=net_ema,
        )

        print('load epoch %s and step %s' % (epoch, step))
        self.step = step  # Resume global step
        return epoch + 1  # Resume from saved epoch

    
    @rank_zero_only
    def save_model(self, epoch, modelmodule):
        """
        Save the current model, optimizer, and EMA states to a checkpoint.
        Only runs on rank 0 in distributed training.
        """
        net = modelmodule.net
        net_without_ddp = modelmodule.net_without_ddp
        net_ema = modelmodule.net_ema
        optimizer = modelmodule.optimizer
        output_dir = self.logger.save_ckpt_dir

        # Save checkpoint periodically or at the last epoch
        print('save ckpt*************************')
        #if (epoch + 1) % self.logger.save_ckpt_freq == 0 or (epoch + 1) == self.max_epochs:
        if (epoch) % self.logger.save_ckpt_freq == 0:
            utils.save_model(
                output_dir=output_dir,
                model=net,
                model_without_ddp=net_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                step=self.step,
                model_ema=net_ema,
                save_meta=False
            )

        # Always save a "latest" checkpoint (may be overwritten each epoch)
        utils.save_model(
            output_dir=output_dir,
            model=net,
            model_without_ddp=net_without_ddp,
            optimizer=optimizer,
            epoch=epoch,
            step=self.step,
            model_ema=net_ema,
            epoch_name='latest',
            save_meta=True
        )