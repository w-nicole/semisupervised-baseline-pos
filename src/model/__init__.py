# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.
# removed irrelevant classes and added new ones

from model.base import Model  # noqa: F401
from model.tagger import Tagger  # noqa: F401
from model.single import Single
from model.joined_ensemble import JoinedEnsemble
from model.on_joined_ensemble import OnJoinedEnsemble
from model.on_split_ensemble import OnSplitEnsemble