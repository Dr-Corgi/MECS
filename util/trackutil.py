import matplotlib.pyplot as plt

class LossTracker():

    def __init__(self):
        self.loss_history = []

    def add(self, loss):
        self.loss_history.append(loss)

    def savefig(self, savepath):
        x = [i for i in range(len(self.loss_history))]
        plt.figure()
        plt.plot(x, self.loss_history)
        plt.savefig(savepath + "loss_track.png")
        with open(savepath+"loss_track.txt",'w') as fout:
            fout.write(str(self.loss_history))


