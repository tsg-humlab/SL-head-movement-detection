class EAFParsingError(Exception):
    def __init__(self, ngt_id, message='Something went wrong with parsing the EAF file'):
        super().__init__(message)
        self.ngt_id = ngt_id
        self.message = message
