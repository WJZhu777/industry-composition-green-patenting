"""Keep commercial-export title overrides outside the release directory."""


class MissingRules(dict):
    def __init__(self, name):
        self.name = name

    def __contains__(self, key):
        raise RuntimeError(
            f"{self.name} requires the original restricted title overrides. "
            "Do not silently substitute an empty dictionary or reclassify the published sample."
        )

    def items(self):
        raise RuntimeError(f"{self.name} requires the original restricted title overrides.")
