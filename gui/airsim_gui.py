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
        # setup gui values
        self.confidence = None
        self.control = None
        self.display_img = None
        self.ts = []
        self.qvals = []
        self.qvals_lb = []
        self.qvals_ub = []

        # setup q vals random gen
        self.cnt = 0 
        self.value = 0
        self.low = low
        self.high = high

    def update_value(self):
        # update display img
        self.display_img = np.random.normal(size=(640,480))

        # update confidence bar
        self.confidence = [np.random.uniform(0, 100)]

        # update control lights
        if np.random.rand() < 0.95:
            self.control = 'agent'
        else:
            self.control = 'human'

        # update Q-vals
        self.value += np.random.uniform(self.low, self.high)
        self.value_std = np.random.rand()
        self.cnt += 1
        self.ts.append(self.cnt)
        self.qvals.append(self.value)
        self.qvals_lb.append(self.value-self.value_std)
        self.qvals_ub.append(self.value+self.value_std)

class GUI(object):
    def __init__(self, env=None):
        # initialize data generation
        try:
            # handles stable baselines DummyVecEnv encapsulation
            self.env = env.envs[0]
        except:
            self.env = env

        # configure which plots to show
        self.SHOW_Q_VALS = False
        self.SHOW_UNCT_TIME = True

        # initialize Qt
        self.app = QtGui.QApplication([])

        # create window
        view = pg.GraphicsView()
        l = pg.GraphicsLayout()
        view.setCentralItem(l)
        view.show()
        view.setWindowTitle('pyqtgraph')
        if self.SHOW_Q_VALS or self.SHOW_UNCT_TIME:
            view.resize(600,950)
        else:
            view.resize(600,800)

        # add plot title
        l.addLabel('The Cycle-of-Learning: AirSim Landing Task', col=0, colspan=3)

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=False)

        # add plot #1 (video from AirSim camera)
        l.nextRow()
        vb = l.addViewBox(lockAspect=True, col=0, colspan=3)
        self.gui_display_img = pg.ImageItem()
        vb.addItem(self.gui_display_img)

        # add bar plot (confidence bar)
        l.nextRow()
        p1 = l.addPlot(title="Current Uncertainty", col=0)
        p1.setLabel('left', '%')
        p1.hideAxis('bottom')
        p1.setRange(xRange=[-.5,1.5])
        p1.setRange(yRange=[0, 100])
        self.confidence_bar = p1.plot(stepMode=True, fillLevel=0, brush=(140,140,170,150))

        # add scatter plot to identify who is controlling (agent or human)
        p2 = l.addPlot(title="Shared Autonomy", col=1, colspan=2)
        p2.addItem(pg.TextItem("AGENT", anchor=(1.75, 2.75), color=(200,200,200,200)))
        p2.addItem(pg.TextItem("HUMAN", anchor=(-0.65, 2.75), color=(200,200,200,200)))
        p2.hideAxis('left')
        p2.hideAxis('bottom')
        p2.setRange(xRange=[-1.85,1.85])
        p2.setRange(yRange=[-1.5,1.5])
        self.control_indicator = pg.ScatterPlotItem(pxMode=False)

        # define initial controlling lights
        lights = []
        lights.append({'pos': (-1,0), 'size': 1, 'brush': (140,140,170,0), 'pen': {'color': 'w', 'width': 1}})
        lights.append({'pos': (1,0), 'size': 1, 'brush': (140,140,170,0), 'pen': {'color': 'w', 'width': 1}})

        # add them to scatter plot structure
        self.control_indicator.addPoints(lights)
        p2.addItem(self.control_indicator)

        # add uncertainty over time plot
        if self.SHOW_UNCT_TIME:
            l.nextRow()
            unct_p = l.addPlot(title="Uncertainty over Time", col=0, colspan=3)
            unct_p.setLabel('bottom', 'Time Step')
            unct_p.setLabel('left', 'Uncertainty (%)')
            unct_p.setRange(yRange=[0, 100])
            self.unct_curve = unct_p.plot(pen=pg.mkPen((140,140,170,255), width=2))
            self.unct_curve_lb = unct_p.plot(pen=pg.mkPen((0,0,0,0), width=2))
            self.unct_curve_ub = unct_p.plot(pen=pg.mkPen((0,0,0,0), width=2))
            unct_fill = pg.FillBetweenItem(self.unct_curve_lb, self.unct_curve_ub, brush = (140,140,170,100))
            unct_p.addItem(unct_fill)

        # add q value plot
        if self.SHOW_Q_VALS:
            l.nextRow()
            qp = l.addPlot(title="Expected Discounted Return", col=0, colspan=3)
            qp.setLabel('bottom', 'Time Step')
            qp.setLabel('left', 'Q(s,a)')
            self.qval_curve = qp.plot(pen=pg.mkPen((140,140,170,255), width=2))
            self.qval_curve_lb = qp.plot(pen=pg.mkPen((0,0,0,0), width=2))
            self.qval_curve_ub = qp.plot(pen=pg.mkPen((0,0,0,0), width=2))
            qval_fill = pg.FillBetweenItem(self.qval_curve_lb, self.qval_curve_ub, brush = (140,140,170,100))
            qp.addItem(qval_fill)
        
        # setup update
        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(50)

        # start the Qt event loop
        self.app.exec_()

    def _update(self):
        # read new values and update plot
        self.gui_display_img.setImage(self.env.display_img)
        self.confidence_bar.setData(np.array([0,1]), self.env.confidence)
        if self.SHOW_Q_VALS:
            # plot qvalues (skips initial 0 values)
            self.qval_curve.setData(self.env.ts[1:], self.env.qvals[1:])
            self.qval_curve_lb.setData(self.env.ts[1:], self.env.qvals_lb[1:])
            self.qval_curve_ub.setData(self.env.ts[1:], self.env.qvals_ub[1:])
        if self.SHOW_UNCT_TIME:
            # plot qvalues (skips initial 0 values)
            self.unct_curve.setData(self.env.ts[1:], self.env.uncts[1:])
            self.unct_curve_lb.setData(self.env.ts[1:], self.env.uncts_lb[1:])
            self.unct_curve_ub.setData(self.env.ts[1:], self.env.uncts_ub[1:])
        
        # control agent/human lights
        lights = []
        if self.env.control == 'agent':
            lights.append({'pos': (-1,0), 'size': 1, 'brush': (140,140,170,150), 'pen': {'color': 'w', 'width': 1}})
            lights.append({'pos': (1,0), 'size': 1, 'brush': (140,140,170,0), 'pen': {'color': 'w', 'width': 1}})
        elif self.env.control == 'human':
            lights.append({'pos': (-1,0), 'size': 1, 'brush': (140,140,170,0), 'pen': {'color': 'w', 'width': 1}})
            lights.append({'pos': (1,0), 'size': 1, 'brush': (140,140,170,150), 'pen': {'color': 'w', 'width': 1}})
        else:
            lights.append({'pos': (-1,0), 'size': 1, 'brush': (140,140,170,0), 'pen': {'color': 'w', 'width': 1}})
            lights.append({'pos': (1,0), 'size': 1, 'brush': (140,140,170,0), 'pen': {'color': 'w', 'width': 1}})
        self.control_indicator.setData(lights)


def main():
    # initialize data generation
    env = DataGenerator()

    # start gui
    gui_thread = threading.Thread(target=GUI, args=(env,))
    gui_thread.start()

    # generate and display new data
    for i in range(1000):
        env.update_value()
        time.sleep(1/50)

    # close everything
    print('[*] Finished experiment.')


if __name__ == "__main__":
    main()