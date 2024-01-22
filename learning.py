import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src.datasets import Communities
from src.learning.callbacks import Callback
from src.learning.data import Data
from src.learning.model import MultiLayerPerceptron

dataset = Communities(continuous=True)
callback = Callback(verbose=False)

if __name__ == '__main__':
    pl.seed_everything(0, workers=True)
    data = Data(x=dataset.input(), y=dataset.target())
    load = DataLoader(dataset=data, batch_size=len(dataset), shuffle=True)
    model = MultiLayerPerceptron(
        dataset=dataset,
        classification=False,
        units=(128, 128),
        penalty=None,
        alpha=None
    )
    train = pl.Trainer(
        max_epochs=5,
        deterministic=True,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[callback],
        logger=False
    )
    train.fit(model=model, train_dataloaders=load)
    print(callback.history.groupby('epoch').mean().drop(columns='batch').transpose())
