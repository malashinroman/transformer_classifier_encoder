from models.clapool import ClaPool
from models.clebert import FillMask
from models.clebert2 import TokenClassifier
from models.fillmaskr_discrete import FillMaskRandDiscrete
from models.fillmaskr_discrete2 import FillMaskRandDiscrete2
from models.gpt2 import FillMaskGPT2
# from models.fillmaskandrandinit import FillMaskRand
from models.fillmaskrandinit import FillMaskRand
from models.fillmaskwnets import FillMaskWithNets
from models.IndexFiller import IndexFiller
from models.SimpleFc import SimpleFc

FILLMASK = FillMask
FILLMASK_GPT2 = FillMaskGPT2
FILLMASK_RAND = FillMaskRand
FILLMASK_RAND_DISCRETE = FillMaskRandDiscrete
FILLMASK_RAND_DISCRETE2 = FillMaskRandDiscrete2
INDEXFILLER = IndexFiller
TOKEN_CLASSIFIER = TokenClassifier
SIMPLE_FC = SimpleFc
FILLMASK_WITH_NETS = FillMaskWithNets
CLAPOOL = ClaPool
