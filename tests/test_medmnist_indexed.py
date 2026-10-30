"""Checks that streamed client images preserve NPZ sample order and values."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from medmnist.info import DEFAULT_ROOT
from torch.utils.data import DataLoader, Subset

from datasets.medmnist_code import (
    _read_indexed_images,
    get_medmnist_client_datasets,
    get_medmnist_datasets,
)


class IndexedMedMNISTTests(unittest.TestCase):
    def test_compressed_rows_match_numpy_indexing(self):
        images = np.arange(13 * 4 * 5 * 3, dtype=np.uint16).reshape(13, 4, 5, 3)
        indices = [12, 0, 7, 7, 1, 11]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.npz"
            np.savez_compressed(path, train_images=images)
            selected, returned_indices = _read_indexed_images(path, "train", indices)

        np.testing.assert_array_equal(selected, images[indices])
        np.testing.assert_array_equal(returned_indices, indices)

    def test_out_of_range_index_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.npz"
            np.savez_compressed(path, train_images=np.zeros((3, 2, 2), dtype=np.uint8))
            with self.assertRaises(IndexError):
                _read_indexed_images(path, "train", [3])

    @unittest.skipUnless((Path(DEFAULT_ROOT) / "pathmnist.npz").is_file(), "PathMNIST cache unavailable")
    def test_cached_pathmnist_batches_match_regular_loader(self):
        train_indices = [89995, 0, 17, 500, 2000, 5]
        test_indices = [7111, 0, 10]
        regular_train, _, regular_test, num_classes = get_medmnist_datasets(
            "pathmnist", download=False, load_val=False
        )
        client_train, client_test, client_classes = get_medmnist_client_datasets(
            "pathmnist", train_indices, test_indices=test_indices
        )

        self.assertEqual(client_classes, num_classes)
        np.testing.assert_array_equal(client_train.imgs, regular_train.imgs[train_indices])
        np.testing.assert_array_equal(client_train.labels, regular_train.labels[train_indices])
        np.testing.assert_array_equal(client_test.imgs, regular_test.imgs[test_indices])
        np.testing.assert_array_equal(client_test.labels, regular_test.labels[test_indices])

        torch.manual_seed(711)
        regular_batch = next(iter(DataLoader(Subset(regular_train, train_indices), batch_size=3, shuffle=True)))
        torch.manual_seed(711)
        client_batch = next(iter(DataLoader(client_train, batch_size=3, shuffle=True)))
        self.assertTrue(torch.equal(regular_batch[0], client_batch[0]))
        self.assertTrue(torch.equal(regular_batch[1], client_batch[1]))


if __name__ == "__main__":
    unittest.main()
