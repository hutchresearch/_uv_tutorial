from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data import GenericDataset
from model.linear import DNN
from runner import Runner
import skeletonkey as sk

from pathlib import Path


@sk.unlock(str(Path(__file__).parent.parent / 'configs/config.yaml'))
def main(cfg):

    # Initialize hyperparameters
    hidden_size = cfg.hidden_size
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr

    # Accelerator is in charge of auto casting tensors to the appropriate GPU device
    accelerator = Accelerator()

    # Initialize the training set and a dataloader to iterate over the dataset
    train_set = GenericDataset()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Get the size of the input and output vectors from the training set
    in_features, out_features = train_set.get_in_out_size()

    # Create the model and optimizer and cast model to the appropriate GPU
    model = DNN(in_features, hidden_size, out_features).to(accelerator.device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Create a runner that will handle
    runner = Runner(
        train_set=train_set,
        train_loader=train_loader,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
    )

    # Train the model
    for _ in range(epochs):

        # Run one loop of training and record the average loss
        train_stats = runner.next()
        print(f"{train_stats}")


if __name__ == "__main__":
    main()
