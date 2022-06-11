# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Removed irrelevant dataset imports.
# Removed LABEL_ID so it's defined in one place.

from dataset.base import Dataset  # noqa: F401
from dataset.tagging import UdPOS
