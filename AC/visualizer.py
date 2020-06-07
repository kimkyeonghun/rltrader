import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc


class Visualizer:

    def __init__(self):
        self.fig = None  # Figure class object in Matplotlib acts like a Canvas
        self.axes = None  # Matxlotlib's Axes class object for drawing charts

    def prepare(self, chart_data):
        # Initialize the canvas and prepare to draw 4 charts
        self.fig, self.axes = plt.subplots(nrows=4, ncols=1, facecolor='w', sharex=True)
        for ax in self.axes:
            # Disable scientific notations that are hard to read
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
        # Chart 1. Daily Chart
        self.axes[0].set_ylabel('Env.')  # y 축 레이블 표시
        # Volume visualization
        x = np.arange(len(chart_data))
        volume = np.array(chart_data)[:, -1].tolist()
        self.axes[0].bar(chart_data["date"], volume, color='b', alpha=0.3)
        # ohlc stands for open, high, low and close, it is a two-dimensional array in this order
        ax = self.axes[0].twinx()
        ax.set_title("Daily Chart")
        ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))
        # Bar chart output to self.axes [0]
        # Positive chart is in red, negative chart is in blue
        candlestick_ohlc(ax, ohlc, colorup='r', colordown='b')

    def plot(self, epoch_str=None, num_epoches=None, epsilon=None, 
            action_list=None, actions=None, num_stocks=None, 
            outvals=None, exps=None, learning=None,
            initial_balance=None, pvs=None):
        x = np.arange(len(actions))  # X-axis data, actions, num_stocks, outvals, exps, and PVs all share the same size
        actions = np.array(actions)  # Agent's actions array
        outvals = np.array(outvals)  # Output array of policy neural network-> matplotlib takes numpy array as input and converts
        pvs_base = np.zeros(len(actions)) + initial_balance  # Initial balance arrangement

        # Chart 2. Agent Status (action, number of holding stocks)
        colors = ['r', 'b']
        for actiontype, color in zip(action_list, colors):
            for i in x[actions == actiontype]:
                self.axes[1].axvline(i, color=color, alpha=0.1)  # Show action with background color
        self.axes[1].plot(x, num_stocks, '-k')  # Draw the number of holding stock

        # Chart 3. Output and Exploration of Policy Neural Networks
        for exp_idx in exps:
            # Draw exploration with yellow background
            self.axes[2].axvline(exp_idx, color='y')
        for idx, outval in zip(x, outvals):
            color = 'white'
            if outval.argmax() == 0:
                color = 'r'  # Red if buying
            elif outval.argmax() == 1:
                color = 'b'  # Blue if selling
            # Draw an action with a red or blue background
            self.axes[2].axvline(idx, color=color, alpha=0.1)
        styles = ['.r', '.b']
        for action, style in zip(action_list, styles):
            # Draw the output of policy neural networks as red and blue dots
            self.axes[2].plot(x, outvals[:, action], style)

        # Chart 4. Portfolio Value
        self.axes[3].axhline(initial_balance, linestyle='-', color='gray')
        self.axes[3].fill_between(x, pvs, pvs_base,
                                  where=pvs > pvs_base, facecolor='r', alpha=0.1)
        self.axes[3].fill_between(x, pvs, pvs_base,
                                  where=pvs < pvs_base, facecolor='b', alpha=0.1)
        self.axes[3].plot(x, pvs, '-k')
        for learning_idx, delayed_reward in learning:
            # Draw learning location in green
            if delayed_reward > 0:
                self.axes[3].axvline(learning_idx, color='r', alpha=0.1)
            else:
                self.axes[3].axvline(learning_idx, color='b', alpha=0.1)

        # Epoch and Exploration Rate
        self.fig.suptitle('Epoch %s/%s (e=%.2f)' % (epoch_str, num_epoches, epsilon))
        # Canvas layout ajustment
        plt.tight_layout()
        plt.subplots_adjust(top=.9)

    def clear(self, xlim):
        for ax in self.axes[1:]:
            ax.cla()  # Erase Green Chart
            ax.relim()  # Initialize limit
            ax.autoscale()  # Reset scale
        # Reset y-axis label
        self.axes[1].set_ylabel('Agent')
        self.axes[2].set_ylabel('AC')
        self.axes[3].set_ylabel('PV')
        for ax in self.axes:
            ax.set_xlim(xlim)  # Reset x-axis limit
            ax.get_xaxis().get_major_formatter().set_scientific(False)  # Disable scientific notation
            ax.get_yaxis().get_major_formatter().set_scientific(False)  # Disable scientific notation
            ax.ticklabel_format(useOffset=False)  # Set x-axis spacing constant
            
    def save(self, path):
        plt.savefig(path)
