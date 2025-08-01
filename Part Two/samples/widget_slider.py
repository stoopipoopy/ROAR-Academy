## This is course material for Introduction to Python Scientific Programming
## Example code: widget_slider.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Create initial plot and values
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5; f0 = 3; delta_f = 0.1; delta_a = 0.1; f1 = 3;
s = (a0 * np.sin(2 * np.pi * f0 * t)) + (a0 * np.sin(2 * np.pi * f1 * t))
l, = plt.plot(t, s, lw=2)
ax.margins(x=0)

# Create two sliders
axcolor = 'lightgoldenrodyellow'
axfreq1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axfreq2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
sfreq1 = Slider(axfreq1, 'Freq1', 0.1, 30.0, valinit=f0, valstep=delta_f)
sfreq2 = Slider(axfreq2, 'Freq2', 0.1, 30.0, valinit=f1, valstep=delta_f)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0, valstep=delta_a)

# slider update actions
def update(val):
    amp = samp.val
    freq = sfreq1.val
    freq2 = sfreq2.val
    l.set_ydata((amp*np.sin(2*np.pi*freq*t)) + amp*np.sin(2*np.pi*freq2*t))
    fig.canvas.draw_idle()

sfreq1.on_changed(update)
sfreq2.on_changed(update)
samp.on_changed(update)

# Create a radio button
rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
l.set_color(radio.value_selected)
# radio button update actions
def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()

radio.on_clicked(colorfunc)

plt.show()
