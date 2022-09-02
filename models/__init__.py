from models.clapool import ClaPool
from models.clebert import FillMask
from models.clebert2 import TokenClassifier
from models.fillmaskr_discrete import FillMaskRandDiscrete

# from models.fillmaskandrandinit import FillMaskRand
from models.fillmaskrandinit import FillMaskRand
from models.fillmaskwnets import FillMaskWithNets
from models.IndexFiller import IndexFiller
from models.SimpleFc import SimpleFc

FILLMASK = FillMask
FILLMASK_RAND = FillMaskRand
FILLMASK_RAND_DISCRETE = FillMaskRandDiscrete
INDEXFILLER = IndexFiller
TOKEN_CLASSIFIER = TokenClassifier
SIMPLE_FC = SimpleFc
FILLMASK_WITH_NETS = FillMaskWithNets
CLAPOOL = ClaPool
