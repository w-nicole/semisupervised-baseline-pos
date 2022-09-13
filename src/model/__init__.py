# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.
# removed irrelevant classes and added new ones

from model.base import Model  # noqa: F401
from model.tagger import Tagger  # noqa: F401
from model.single import Single
from model.joined_ensemble import JoinedEnsemble
from model.split_ensemble import SplitEnsemble