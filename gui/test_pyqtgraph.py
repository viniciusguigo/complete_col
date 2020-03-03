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
        view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(100,100,100))
        view.setCentralItem(l)
        view.show()
        view.setWindowTitle('pyqtgraph')
        view.resize(800,600)

        # add plot title
        l.addLabel('The Cycle-of-Learning Experiment', col=0, colspan=1)
        l.nextRow()

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # add plot #1
        p1 = l.addPlot(title="Total Reward per Episode")
        p1.setLabel('bottom', 'Episode')
        p1.setLabel('left', 'Total Reward')
        self.curve1 = p1.plot(pen=pg.mkPen('y', width=2))

        # add plot #2 on the next line
        l.nextRow()
        p2 = l.addPlot(title="Expect Discounted Return")
        p2.setLabel('bottom', 'Time Step')
        p2.setLabel('left', 'Q(s,a)')
        self.curve2 = p2.plot(pen=pg.mkPen((80,0,0), width=3))

        # # add bar plot besides plot #2
        # bg = pg.BarGraphItem(x=1, height=100, width=0.3, brush='g')
        # p3 = l.addPlot(title="Confidence")
        
        # setup update
        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(50)

        # start the Qt event loop
        self.app.exec_()

    def _update(self):
        # read new values and update plot
        self.curve1.setData(self.datagen.xvals, self.datagen.yvals)
        self.curve2.setData(self.datagen.xvals, self.datagen.yvals)

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