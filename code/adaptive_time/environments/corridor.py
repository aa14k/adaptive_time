
class Corridor:
    """A linear corridor, where we start from one end and try to reach the other."""
    def __init__(self, length=5, horizon_sec=200.0, dt_sec=1.0):
        self._length = length
        self.horizon_sec = horizon_sec
        self.dt_sec = dt_sec
        self.horizon_num = int(horizon_sec / dt_sec)
        self.reset()

    def reset(self):
        self.pos = 0
        self.done = False
        self.h_sec = 0
        self.h_num = -1
        return self.pos

    def step(self, a):
        if a not in (0, 1, 2):
            raise ValueError(f"Invalid action: {a}.")

        self.h_sec += self.dt_sec
        self.h_num += 1

        action = a - 1
        self.pos += action * self.dt_sec

        reward = -1.0
        if self.pos <= 0:
            self.pos = 0
        elif self.pos >= self._length-1:
            self.pos = self._length-1
            reward = 0.0
        
        if self.h_sec + self.dt_sec > self.horizon_sec:
            self.done = True
        
        return reward, self.pos, (self.h_sec, self.h_num), self.done
        
