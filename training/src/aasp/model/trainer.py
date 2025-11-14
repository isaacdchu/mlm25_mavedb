import torch
import time

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        num_epochs=10,
        save_path="output/baseline_model_best.pth",
        device="cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.device = device

    def run(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                print("Best val loss improved, saving model...")
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        total_batches = len(self.train_loader)
        last_print = time.time()
        for batch_idx, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            
            # Dynamic feature assignment based on X.shape[1]
            if X.shape[1] >= 6:
                distance = X[:, 0:1]
                biotype = X[:, 1].long()
                ref_aa = X[:, 2].long()
                alt_aa = X[:, 3].long()
                scoreset = X[:, 4].long()
                consequence = X[:, 5:]
            else:
                # Adjust as per your actual field order and count, minimum is just distance
                distance = X[:, 0:1]
                biotype = ref_aa = alt_aa = scoreset = None
                consequence = None

            self.optimizer.zero_grad()
            # Only pass features that are available
            y_hat = self.model(
                distance,
                biotype=biotype,
                ref_aa=ref_aa,
                alt_aa=alt_aa,
                scoreset=scoreset,
                consequence=consequence
            )
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            # Progress heartbeat every 60s or on last batch
            now = time.time()
            if now - last_print > 60 or batch_idx == total_batches-1:
                pct = 100.0 * (batch_idx + 1) / total_batches
                print(f"   training... {pct:5.1f}% | "
                      f"batch {batch_idx+1}/{total_batches} | "
                      f"loss={loss.item():.4f}")
                last_print = now
        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (X, y) in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                if X.shape[1] >= 6:
                    distance = X[:, 0:1]
                    biotype = X[:, 1].long()
                    ref_aa = X[:, 2].long()
                    alt_aa = X[:, 3].long()
                    scoreset = X[:, 4].long()
                    consequence = X[:, 5:]
                else:
                    distance = X[:, 0:1]
                    biotype = ref_aa = alt_aa = scoreset = None
                    consequence = None
                y_hat = self.model(
                    distance,
                    biotype=biotype,
                    ref_aa=ref_aa,
                    alt_aa=alt_aa,
                    scoreset=scoreset,
                    consequence=consequence
                )
                loss = self.loss_fn(y_hat, y)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)
