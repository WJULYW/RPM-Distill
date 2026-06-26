
from data.equipleth_dataset import PairedVideoRFWindowDataset


class DatasetFactory:
    """Create datasets required by the current distillation pipeline."""

    @staticmethod
    def create_paired_distill_dataset(target_domain: str, **kwargs):
        """
        Create paired distillation dataset with aligned RGB/RF/PPG windows.
        """
        target_domain = target_domain.lower()
        if target_domain == "equipleth":
            return PairedVideoRFWindowDataset(**kwargs)
        raise ValueError(
            f"Unsupported target domain '{target_domain}' in the cleaned project. "
            "Only 'equipleth' is kept for this main.py entry pipeline."
        )
