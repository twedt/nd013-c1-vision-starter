import argparse
from collections import Counter
import pathlib
from typing import Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


from utils import get_module_logger, get_object_distribution, get_dataset

def split(
    dataset:tf.data.Dataset,
    source_dir:pathlib.Path,
    destination_dir:pathlib.Path,
    train_split:float=0.85,
    val_split:float=0.09,
    test_split:float=0.06,
    seed_state:int=0,
):
    """
    Create train, val, and test folders with soft links to source labeled data.

    Args:
        dataset: tf dataset sample to include all source files.
        source_dir: path to tfrecord source files
        destination_dir: base path for split sub folders train / val / test.
        train_split: percentage of source data used for training.
        val_split: percentage of source data used for validation.
        test_split: percentage of source data used for test.
        seed_state: random seed.
    """
 
    train_path = pathlib.Path(destination_dir, 'train')
    if not train_path.exists():
        train_path.mkdir()
    
    test_path = pathlib.Path(destination_dir, 'test')
    if not test_path.exists():
        test_path.mkdir()
    
    val_path = pathlib.Path(destination_dir, 'val')
    if not val_path.exists():
        val_path.mkdir()
   
    if train_split + test_split + val_split > 1.0 + 1e-6 or \
         train_split + test_split + val_split < 1.0 - 1e-6:
        raise ValueError("train, test, and val splits must sum to 1.0")
    
    (
        _, 
        _, 
        source_id_obj_count,
        source_id_frame_count,
    ) =  get_object_distribution(dataset)

    (
        train_files,
        val_files,
        test_files,
    ) =_stratify_dataset(
        source_id_obj_count,
        source_id_frame_count,
        train_split,
        val_split,
        test_split,
        seed_state,
    )
        
    _create_symbolic_link_to_source(train_files, source_dir, train_path)
    _create_symbolic_link_to_source(val_files, source_dir, val_path)
    _create_symbolic_link_to_source(test_files, source_dir, test_path)


def _stratify_dataset(
    source_id_obj_count: Counter,
    source_id_frame_count: Counter,    
    train_split: float,
    val_split: float,
    test_split: float,
    seed_state: int,
    max_density_bin: int=100,
    num_bins: int=5,
) -> Tuple[Set, Set, Set]:
    """Create dataset splits based on average object density across source files.
    
        Args:
            source_id_frame_count: contains source file frame counts.
            source_id_obj_count: contains source file label object count.
            max_density_bin: maximum number of objects per frame.
            num_bins: number of object density bins for stratification.

        Returns:
            Sequence containing train, val, and test source files.
    """
    
    # Initialize variables.
    source_idx = np.expand_dims(np.arange(len(source_id_frame_count)), axis=1)
    source_ids = {}
    density_bins = np.linspace(0,max_density_bin,num_bins+1)
    source_avg_object_density_bin = np.zeros([len(source_id_frame_count), num_bins])

    # Generate density stratification "labels".
    for i, key in enumerate(sorted(source_id_frame_count)):
        source_ids[i] = key
        source_avg_object_density = source_id_obj_count[key] / source_id_frame_count[key]
        
        # Decompose average into one-hot encoding across density bins.
        hist, _ = np.histogram(source_avg_object_density, density_bins)
        source_avg_object_density_bin[i][:] = hist

    # Generate density stratified train split.
    x_train, x_remaining = train_test_split(
        source_idx,
        train_size=train_split, 
        random_state=seed_state, 
        stratify=source_avg_object_density_bin
    )

    # Generate density stratified val/test split
    x_val, x_test = train_test_split(
        x_remaining,
        train_size=(val_split / (val_split + test_split)),
        random_state=seed_state,
        stratify=source_avg_object_density_bin[np.squeeze(x_remaining)]
    )

    # Map source file indicies to filenames.
    x_train_files = set([source_ids[idx] for idx in np.squeeze(x_train)])
    x_val_files = set([source_ids[idx] for idx in np.squeeze(x_val)])
    x_test_files = set([source_ids[idx] for idx in np.squeeze(x_test)])

    return (x_train_files, x_val_files, x_test_files)

def _create_symbolic_link_to_source(
    source_files:Set[pathlib.Path],
    source_dir:pathlib.Path,
    destination_dir:pathlib.Path,
):
    """Create symbolic links in destination directory to source file.
    
    Args:
        source_files: sequence of source file names.
        source_dir: location of source files.
        destination_dir: destination directory for symbolic links.
    """

    for source_file in source_files:
        symbolic_link_path = pathlib.Path(destination_dir, source_file)
        symbolic_link_path.symlink_to(source_dir.joinpath(source_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    parser.add_argument('-ss', required=False,
                        help='number of samples to extract from the dataset',
                        type=int,
                        default=10000)
    
    args = parser.parse_args()
    logger = get_module_logger(__name__)

    source_dir = pathlib.Path(args.source)
    if not source_dir.exists():
        raise NameError("Source directory does not exist!")
    
    all_source_files = source_dir.joinpath('*.tfrecord')
    dataset = get_dataset(str(all_source_files)).take(args.ss)
   
    destination_dir = pathlib.Path(args.destination)
    if not destination_dir.exists():
        raise NameError("Destination directory does not exist!")
    
    logger.info('Creating splits...')
    split(dataset, source_dir, destination_dir)