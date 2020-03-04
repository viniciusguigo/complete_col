"""
GUI designed for the Cycle-of-Learning experiments using Microsoft AirSim.

The main goals are to display in real time to the user:
    - agent's confidence level;
    - who is controlling the vehicle (agent or human);
    - camera feed;
    - [optional] joystick inputs;
    - [optional] Q value plot for each state-action pair.

"""
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
        view.resize(640,600)

        # add plot title
        l.addLabel('The Cycle-of-Learning: AirSim Landing Task', col=0, colspan=3)

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # add plot #1 (video from AirSim camera)
        l.nextRow()
        vb = l.addViewBox(lockAspect=True, col=0, colspan=3)
        self.img = pg.ImageItem()
        vb.addItem(self.img)

        # add bar plot (confidence bar)
        l.nextRow()
        p1 = l.addPlot(title="Confidence", col=0)
        p1.setLabel('left', '%')
        p1.hideAxis('bottom')
        p1.setRange(xRange=[-.5,1.5])
        p1.setRange(yRange=[0, 100])
        self.bar = p1.plot(stepMode=True, fillLevel=0, brush=(0,255,0,150))

        # add scatter plot to identify who is controlling (agent or human)
        p2 = l.addPlot(title="Shared Autonomy", col=1, colspan=2)
        p2.addItem(pg.TextItem("AGENT", anchor=(1.75, 2.75), color=(200,200,200,200)))
        p2.addItem(pg.TextItem("HUMAN", anchor=(-0.65, 2.75), color=(200,200,200,200)))
        p2.hideAxis('left')
        p2.hideAxis('bottom')
        p2.setRange(xRange=[-2,2])
        p2.setRange(yRange=[-1,1])
        self.control = pg.ScatterPlotItem(pxMode=False)

        # define initial controlling lights
        lights = []
        lights.append({'pos': (-1,0), 'size': 1, 'brush': (0,255,0,0), 'pen': {'color': 'w', 'width': 1}})
        lights.append({'pos': (1,0), 'size': 1, 'brush': (0,255,0,0), 'pen': {'color': 'w', 'width': 1}})

        # add them to scatter plot structure
        self.control.addPoints(lights)
        p2.addItem(self.control)
        
        # setup update
        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(50)

        # start the Qt event loop
        self.app.exec_()

    def _update(self):
        # read new values and update plot
        self.img.setImage(np.random.normal(size=(640,480)))
        self.bar.setData(np.array([0,1]), [np.random.uniform(0, 100)])
        
        # control agent/human lights
        lights = []
        if np.random.rand() < 0.9:
            lights.append({'pos': (-1,0), 'size': 1, 'brush': (0,255,0,150), 'pen': {'color': 'w', 'width': 1}})
            lights.append({'pos': (1,0), 'size': 1, 'brush': (0,255,0,0), 'pen': {'color': 'w', 'width': 1}})
        else:
            lights.append({'pos': (-1,0), 'size': 1, 'brush': (0,255,0,0), 'pen': {'color': 'w', 'width': 1}})
            lights.append({'pos': (1,0), 'size': 1, 'brush': (0,255,0,150), 'pen': {'color': 'w', 'width': 1}})
        self.control.setData(lights)


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