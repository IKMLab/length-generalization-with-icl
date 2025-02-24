import math


class Curriculum:

    def __init__(self, args):
        self.n_dims_truncated = args["dims"]["start"]
        self.n_points = args["points"]["start"]
        self.n_dims_schedule = args["dims"]
        self.n_points_schedule = args["points"]
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated,
            self.n_dims_schedule,
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule["interval"] == 0:
            var += schedule["increment"]

        return min(var, schedule["end"])
