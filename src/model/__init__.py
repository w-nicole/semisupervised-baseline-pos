# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.
# removed irrelevant classes and added new ones

from model.base import Model  # noqa: F401
from model.tagger import Tagger  # noqa: F401
from model.base_tagger import BaseTagger
from model.latent_to_pos import LatentToPOS
from model.latent_to_pos_cross_target import LatentToPOSCross
from model.latent_to_pos_cross_with_word import LatentToPOSCrossWord