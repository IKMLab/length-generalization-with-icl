import os

import yaml
from typing import IO


class Loader(yaml.SafeLoader):

    def __init__(self, stream: IO) -> None:
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def construct_include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, "r") as f:
            return yaml.load(f, Loader)
