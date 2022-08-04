from models.clapool import ClaPool
from models.clebert import FillMask
from models.clebert2 import TokenClassifier

# from models.fillmaskandrandinit import FillMaskRand
from models.fillmaskrandinit import FillMaskRand
from models.fillmaskwnets import FillMaskWithNets
from models.IndexFiller import IndexFiller
from models.SimpleFc import SimpleFc

FILLMASK = FillMask
FILLMASK_RAND = FillMaskRand
INDEXFILLER = IndexFiller
TOKEN_CLASSIFIER = TokenClassifier
SIMPLE_FC = SimpleFc
FILLMASK_WITH_NETS = FillMaskWithNets
CLAPOOL = ClaPool
