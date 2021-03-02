from common import google_cloud


class GANTrainParameters():
    def __init__(self):
        self.num_epochs = 2000
        self.batch_size = 10000
        self.num_steps = 1
        self.lr_d = 0.01
        self.lr_g = 0.001

        if not google_cloud:
            self.batch_size = 1


training_param = GANTrainParameters()
