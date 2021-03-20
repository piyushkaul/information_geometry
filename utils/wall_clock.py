import time
class WallClock():

    def __init__(self):
        self.start_time = time.time()

    def elapsed_time(self):
        curr_time = time.time()
        return curr_time - self.start_time
