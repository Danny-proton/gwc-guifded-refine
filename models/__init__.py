from models.gwcnet import GwcNet_G, GwcNet_GC
from models.loss import model_loss
from models.loss import silog_loss
from models.gwcnet_8 import Gwcnet_8
from models.gwcnet_dn import Gwcnet_dn

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-gc": GwcNet_GC,
    "gwcnet_8":Gwcnet_8,
    "gwcnet_dn":Gwcnet_dn
}
