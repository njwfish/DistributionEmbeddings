import os
import torch
import pandas as pd
import time
from tqdm import tqdm
from utils.hash_utils import get_output_dir

class MinimalTrainer:
    def __init__(
        self,
        num_epochs=100,
        log_interval=10,
        eval_interval=5,
        early_stopping=True,
        patience=10,
        use_tqdm=True,
        csv_logger=True,
    ):
        # init params!
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.early_stopping = early_stopping
        self.patience = patience
        self.use_tqdm = use_tqdm
        self.csv_logger = csv_logger
        if csv_logger:
            self.csv_logger = CSVLogger()

        self.best_loss = float('inf')
        self.no_improve_count = 0

    def train(
        self,
        encoder,
        generator,
        dataloader,
        optimizer,
        scheduler=None,
        device=None,
        output_dir='./outputs',
        config=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config is not None:
            output_dir = get_output_dir(config, base_dir=output_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)

        encoder.to(device)
        generator.model.to(device)

        for epoch in range(self.num_epochs):
            encoder.train()
            generator.model.train()
            epoch_losses = []

            pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{self.num_epochs}") if self.use_tqdm else dataloader

            for i, batch in enumerate(pbar):
                x = batch['samples'].to(device)
                optimizer.zero_grad()
                z = encoder(x)
                loss = generator.loss(x.view(-1, *x.shape[2:]), z)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                if i % self.log_interval == 0:
                    print(f"epoch {epoch+1}, batch {i}, loss {loss.item():.4f}")
                    if self.use_tqdm:
                        pbar.set_postfix(loss=f"{loss.item():.4f}")

            if scheduler:
                scheduler.step()

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"epoch {epoch+1} done! avg loss: {avg_loss:.4f}")

            if (epoch + 1) % self.eval_interval == 0:
                val_loss = self._evaluate(encoder, generator, dataloader, device)
                print(f"eval loss: {val_loss:.4f}")

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.no_improve_count = 0
                    print("new best! saving :)")

                    torch.save({
                        'epoch': epoch + 1,
                        'encoder_state_dict': encoder.state_dict(),
                        'generator_state_dict': generator.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                    }, os.path.join(output_dir, 'best_model.pt'))

                else:
                    self.no_improve_count += 1
                    print(f"no improve ({self.no_improve_count})")

                if self.early_stopping and self.no_improve_count >= self.patience:
                    print("early stopping!")
                    break

        # write logs after training ends!
        if self.csv_logger:
            self.csv_logger.write(encoder, generator, dataloader, output_dir, device)

        return output_dir, None

    def _evaluate(self, encoder, generator, dataloader, device):
        # quick eval loop!
        encoder.eval()
        generator.model.eval()
        total_loss, n = 0, 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch['samples'].to(device)
                z = encoder(x)
                loss = generator.loss(x.view(-1, *x.shape[2:]), z)
                total_loss += loss.item()
                n += 1

        return total_loss / n
    

class CSVLogger:
    def __init__(self, filename='log.csv'):
        self.filename = filename

    def write(self, encoder, generator, dataloader, output_dir, device):
        encoder.eval()
        generator.model.eval()

        data = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch['samples'].to(device)
                m = batch['metadata']#.to(device)
                z = encoder(x)
                y = generator.sample(z, num_samples=10**3)

                for i in range(len(x)):
                    input_item = x[i].cpu().numpy().flatten().tolist()
                    metadata_item = m[1][i]
                    embedding_item = z[i].cpu().numpy().flatten().tolist()
                    output_item = y[i].cpu().numpy().flatten().tolist()

                    data.append({
                        'input': input_item,
                        'metadata': metadata_item,
                        'embedding': embedding_item,
                        'output': output_item
                    })

        df = pd.DataFrame(data)
        out_path = os.path.join(output_dir, self.filename)
        df.to_csv(out_path, index=False)

        print(f"csv saved to {out_path} 📝🎉")