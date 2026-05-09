from tensprov.queries import forward, backward


class ProvenanceEngine:
    def __init__(self, steps, input_dim=1, output_dim=0):
        self.steps = steps
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _forward_step(self, step, current):
        # If step is an Operation (has .forward method)
        if hasattr(step, "forward"):
            return step.forward(current, input_dim=self.input_dim, output_dim=self.output_dim)
        # Otherwise, it's a raw index
        return forward(
            step,
            input_dim=self.input_dim,
            input_ids=current,
            output_dim=self.output_dim,
        )

    def _backward_step(self, step, current):
        if hasattr(step, "backward"):
            return step.backward(current, input_dim=self.input_dim, output_dim=self.output_dim)
        return backward(
            step,
            output_ids=current,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

    def forward(self, records):
        current = set(records)
        for step in self.steps:
            current = self._forward_step(step, current)
        return current

    def backward(self, records):
        current = set(records)
        for step in reversed(self.steps):
            current = self._backward_step(step, current)
        return current

    def forward_with_sizes(self, records):
        current = set(records)
        sizes = [len(current)]

        for step in self.steps:
            current = self._forward_step(step, current)
            sizes.append(len(current))

        return current, sizes
