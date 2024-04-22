import numpy as np


class Util:
    def _generate_values(self, start, end, count, round_to_int=False):
        values = np.linspace(start, end, count)

        if round_to_int:
            values = np.round(values).astype(int)

        return values.tolist()