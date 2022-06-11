# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.
# removed irrelevant classes

from model.base import Model  # noqa: F401
from model.tagger import Tagger  # noqa: F401
# below: all added
from model.base_vae import BaseVAE
from model.vae import VAE