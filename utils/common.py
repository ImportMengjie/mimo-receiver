class AvgLoss:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def add(self, num):
        self.sum += num
        self.count += 1
        self.avg = self.sum/self.count
