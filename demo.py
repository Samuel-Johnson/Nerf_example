import torch
import lightning as L

from nerf_example.network import TinyNet
from nerf_example.render import render
from nerf_example.datasets import create_dataset

class Demo(L.LightningModule):

    def __init__(self, network):
        super().__init__()
        self.model = network

    def training_step(self, batch, batch_idx):
        rays_o, rays_d, images = batch

        bounds = torch.ones(*rays_o.shape[:-1])
        bounds = bounds.to(device=self.device)
        bounds = [2.0 * bounds, 6.0 * bounds]
        
        c_hat = render(self.model, rays_o, rays_d, bounds)

        #compute loss
        loss = torch.sum(torch.square(c_hat - images))
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        rays_o, rays_d, images = batch
        assert rays_o.shape[0] == 1, f"validation set has shape {rays_o.shape}"

        bounds = torch.ones(*rays_o.shape[:-1])
        bounds = bounds.to(device=self.device)
        bounds = [2.0 * bounds, 6.0 * bounds]
        
        c_hat = render(self.model, rays_o, rays_d, bounds) # [1, H, W, 3]
        c_hat = c_hat.squeeze() # remove the first dimension

        loss = torch.sum(torch.square(c_hat - images))
        self.log("train_loss", loss)
        
        
        self.logger.experiment.add_image(f"image_at_{self.current_epoch}", c_hat.permute(2,0,1))
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        return optimizer

if __name__ == "__main__":
    batch_size = 1
    training_dataset, val_dataset = create_dataset(hold_out=True)
    train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    network = TinyNet()

    lit_module = Demo(network=network)
    
    trainer = L.Trainer(limit_train_batches=100, max_epochs=25)
    trainer.fit(model=lit_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
