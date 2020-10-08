import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, name, learner, axes, legend_title=None):
        self.name = name
        self.learner = learner
        self.x_axis = axes['x']
        self.y_axis = axes['y']
        self.legend_title = legend_title

    def add_plot(self, x, y, label=None, marker='.', alpha=1.0):
        if label is None:
            plt.plot(x, y, linestyle='-', marker=marker, alpha=alpha)
        else:
            plt.plot(x, y, linestyle='-', marker=marker, label=label, alpha=alpha)

    def find_max(self, x, y, label):
        y = list(y)
        max_acc, offset = max(y), max(x)*.0125
        i = y.index(max_acc)
        plt.axvline(x=x[i], label='{}={:.4f} (max)'.format(self.y_axis, y[i], label), color='g')
        plt.text(x=x[i]+offset, y=100, s='{:.1f}'.format(max_acc))

    def find_max_int(self, x, y, label):
        y = list(y)
        max_acc, offset = max(y), max(x)*.0125
        i = y.index(max_acc)
        plt.axvline(x=x[i], label='{}={} ({})'.format(self.x_axis, x[i], label), color='g')
        plt.text(x=x[i]+offset, y=100, s='{:.1f}'.format(max_acc))

    def find_min(self, x, y, label, top=True):
        y = list(y)
        min_mse, x_offset = min(y), max(x)*.0125
        y_offset = (max(y)/2.) + min(y) if top else (max(y)/2.) - min(y)
        i = y.index(min_mse)
        plt.axvline(x=x[i], label='{}={:.4f} (min)'.format(self.y_axis, y[i], label), color='g')

    def find_min_int(self, x, y, label, top=True):
        y = list(y)
        min_mse, x_offset = min(y), max(x)*.0125
        y_offset = (max(y)/2.) + min(y) if top else min(y) - (max(y)/2.) 
        i = y.index(min_mse)
        plt.axvline(x=x[i], label='{}={} ({})'.format(self.y_axis, y[i], label), color='g')

    def save(self, loc='best', framealpha=.8, top_limit=None):
        if top_limit is not None:
            plt.ylim(top=top_limit)
        plt.xlabel(self.x_axis)
        plt.ylabel(self.y_axis)
        plt.title(self.name)
        if self.legend_title is not None:
            plt.legend(loc=loc, framealpha=framealpha, title=self.legend_title)
        else:
            plt.legend(loc=loc, framealpha=framealpha)
        plt.savefig('images/{}/{}'.format(self.learner, self.name))
        plt.close()