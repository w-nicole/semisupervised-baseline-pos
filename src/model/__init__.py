# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

from model.aligner import Aligner  # noqa: F401
from model.base import Model  # noqa: F401
from model.classifier import Classifier  # noqa: F401
from model.dependency_parser import DependencyParser  # noqa: F401
from model.tagger import Tagger  # noqa: F401
# below: all added
from model.encoder_decoder import EncoderDecoder 