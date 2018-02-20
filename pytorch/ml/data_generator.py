class DataGenerator:
    """Make a given <data_loader> go on and on forever"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._iterator = iter(self.data_loader)

    def __next__(self):
        return self.next()

    def next(self):
        try:
            return self._iterator.next()
        except StopIteration:
            self._iterator = iter(self.data_loader)
            return self._iterator.next()
