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
        l = pg.GraphicsLayout()
        view.setCentralItem(l)
        view.show()
        view.setWindowTitle('pyqtgraph')
        view.resize(800,600)

        # add plot title
        l.addLabel('The Cycle-of-Learning Experiment', col=0, colspan=4)

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # add plot #1
        l.nextRow()
        p1 = l.addPlot(title="Total Reward per Episode", col=0, colspan=4)
        p1.setLabel('bottom', 'Episode')
        p1.setLabel('left', 'Total Reward')
        self.curve1 = p1.plot(pen=pg.mkPen('y', width=2))

        # add plot #2 on the next line
        l.nextRow()
        p2 = l.addPlot(title="Expect Discounted Return", col=0, colspan=3)
        p2.setLabel('bottom', 'Time Step')
        p2.setLabel('left', 'Q(s,a)')
        self.curve2 = p2.plot(pen=pg.mkPen((80,0,0), width=3))

        # add bar plot besides plot #2
        p3 = l.addPlot(title="Confidence", col=3)
        p3.setLabel('left', '%')
        p3.hideAxis('bottom')
        p3.setRange(xRange=[-.5,1.5])
        p3.setRange(yRange=[0, 100])
        # p3.plot(np.array([0,1]), np.array([100]), stepMode=True, fillLevel=0, brush=(0,255,0,150))
        self.curve3 = p3.plot(stepMode=True, fillLevel=0, brush=(0,255,0,150))
        
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
        self.curve3.setData(np.array([0,1]), [np.random.uniform(0, 100)])

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