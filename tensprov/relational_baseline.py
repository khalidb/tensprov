class RelationalProvenance:
    def __init__(self, dims):
        self.dims = dims
        self.rows = []

    def add_many(self, coords):
        self.rows.extend(coords)

    def project(self, source_dim, source_ids, target_dim):
        result = set()

        for row in self.rows:
            if row[source_dim] in source_ids:
                result.add(row[target_dim])

        return result
