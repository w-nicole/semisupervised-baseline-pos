# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.
# removed irrelevant classes and added new ones

from model.base import Model  # noqa: F401
from model.tagger import Tagger  # noqa: F401
from model.base_tagger import BaseTagger
from model.latent_base import LatentBase
from model.changing_target import ChangingTarget
from model.fixed_target import FixedTarget