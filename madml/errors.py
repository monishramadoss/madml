class InvalidArguementError(Exception):
    def __init__(self, message, error):
        super(InvalidArguementError, self).__init__(message)
        self.error = error