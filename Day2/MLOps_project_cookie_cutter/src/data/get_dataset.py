import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
from torchvision.datasets import MNIST
from torchvision import transforms
import os
import shutil
from pathlib import Path
import time


@click.command()
@click.option('--process_data', default=True, help='Apply  to the raw dataset')
def main(process_data):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Collecting the dataset')
    download = True
    transformations = None
    # Assume data
    if Path('./data/test/MNIST').exists():
        print("Data already downloaded.")
        download = False

    if process_data:
        transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,)),
                                              ])

    while not Path('./data/MNIST/processed/test.pt').exists():
        test = MNIST("./data", train=False, download=download, transform=transformations)
        time.sleep(1)
        print("Downloaded Test data")
    while not Path('./data/MNIST/processed/training.pt').exists():
        train = MNIST("./data", train=True, download=download, transform=transformations)
        time.sleep(1)
        print("Downloaded Test data")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
