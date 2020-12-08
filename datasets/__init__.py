from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .cityscape_dataset import CityscapeDatset
from .middlebury import Middlebury
from .eth3d import ETH3Ddataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "cityscape":CityscapeDatset,
    "middlebury":Middlebury,
    "eth3d":ETH3Ddataset
}
