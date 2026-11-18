"""Report missing restricted title classifications."""


class MissingRules(dict):
    def __init__(self, name):
        self.name = name

    def __contains__(self, key):
        raise RuntimeError(f"{self.name} requires the original restricted title overrides.")

    def items(self):
        raise RuntimeError(f"{self.name} requires the original restricted title overrides.")
