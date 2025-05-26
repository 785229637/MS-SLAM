from datasets.gradslam_datasets.replica import ReplicaDataset
from datasets.gradslam_datasets.scannet import ScannetDataset
from datasets.gradslam_datasets.tum import TUMDataset


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)

    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")