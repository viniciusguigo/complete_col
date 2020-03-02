from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time


class DataGenerator(object):
    def __init__(self, low=-1, high=1):
        self.cnt = 0
        self.value = 0
        self.low = low
        self.high = high
        self.xvals = []
        self.yvals = []

    def update_value(self):
        self.value += np.random.uniform(self.low, self.high)
        self.cnt += 1
        self.xvals.append(self.cnt)
        self.yvals.append(self.value)
        # print(f'X = {self.cnt} | Y = {self.value}')

class GUI(object):
    def __init__(self, datagen=None):
        # initialize data generation
        self.datagen = datagen

        # initialize Qt
        self.app = QtGui.QApplication([])

        # create window
        win = pg.GraphicsWindow(title="Reinforcement Learning Experiment")
        win.resize(1000,600)

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # add plot
        p6 = win.addPlot(title="Environment XYZ")
        p6.setLabel('bottom', 'Episode')
        p6.setLabel('left', 'Total Reward')
        self.curve = p6.plot(pen='y')
        
        # setup update
        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(50)

        # start the Qt event loop
        self.app.exec_()

    def _update(self):
        # read new values and update plot
        self.curve.setData(self.datagen.xvals, self.datagen.yvals)

def init_gui(datagen):
    # Initialize GUI in a separate thread (not the main thread)
    display = GUI(datagen)

def main():
    # initialize data generation
    datagen = DataGenerator()

    # start gui
    gui_thread = threading.Thread(target=init_gui, args=(datagen,))
    gui_thread.start()

    # generate and display new data
    for i in range(1000):
        datagen.update_value()
        time.sleep(1/50)

    # close everything
    print('[*] Finished experiment.')


if __name__ == "__main__":
    main()