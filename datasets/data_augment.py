from torch.utils.data import Dataset
from torchvision import transforms
from datasets.multi_dataset import MultiDataset

class AugmentedAsciiDataset(Dataset):
    """
    Wraps a dataset to apply on-the-fly data augmentation for training.

    This class takes an existing dataset and applies a series of transformations
    to the images only when in 'train' mode. For validation/testing, it only
    converts the image to a tensor.
    """

    def __init__(self, original_dataset: MultiDataset, is_train=True, augment_params: dict = None):
        """
        Initializes the wrapper and the transformation pipeline.

        Args:
            original_dataset (Dataset): The source dataset (e.g., AsciiC64_Dataset).
            is_train (bool): If True, applies the full augmentation pipeline.
            augment_params (dict): A dictionary of augmentation settings.
        """
        self.original_dataset = original_dataset
        self.is_train = is_train

        # Default transformation (no augmentation)
        transform_list = [transforms.ToPILImage(), transforms.ToTensor()]

        params = augment_params
        if self.is_train and params:
            # If training and params are provided, build the augmentation pipeline.
            aug_transform_list = [transforms.ToPILImage()]

            if "RandomRotation_degrees" in params:
                aug_transform_list.append(transforms.RandomRotation(
                    degrees=params["RandomRotation_degrees"],
                    fill=params.get("RandomAffine_fill", 0)
                ))

            if "RandomAffine_translate_x" in params or "RandomAffine_translate_y" in params:
                translate_x = params.get("RandomAffine_translate_x", 0.0)
                translate_y = params.get("RandomAffine_translate_y", 0.0)
                aug_transform_list.append(transforms.RandomAffine(
                    degrees=0,  # Rotation is handled above
                    translate=(translate_x, translate_y),
                    fill=params.get("RandomAffine_fill", 0)
                ))

            if "RandomHorizontalFlip_p" in params:
                aug_transform_list.append(transforms.RandomHorizontalFlip(
                    p=params["RandomHorizontalFlip_p"]
                ))

            aug_transform_list.append(transforms.ToTensor())
            transform_list = aug_transform_list

        self.transform = transforms.Compose(transform_list)

    def get_class_counts(self):
        return self.original_dataset.get_class_counts()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.original_dataset)

    def __getitem__(self, idx):
        """
        Retrieves a sample, applies the appropriate transformation, and returns it.
        """
        image_block, label = self.original_dataset[idx]

        # The transform pipeline expects a PIL image.
        image_block = self.transform(image_block)

        return image_block, label
