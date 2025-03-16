import math
from math import sin, cos, exp
import numpy as np
import pandas as pd
from scipy import signal

RT2 = math.sqrt(2)
PI = math.pi

# Technical Analysis

def TrueRange(h, l, c):
    """ Standard TA true range
    """
    s = pd.concat([(h-l), (h-c.shift()), (l-c.shift())], axis=1).max(axis=1)
    return s

# Statistical

def ZScore(s, length, min_periods=None):
    """ Rolling z-score
    """
    return (s - s.rolling(length, min_periods=min_periods).mean().shift())/s.rolling(length, min_periods=min_periods).std().shift()

def RMS(src, length):
    """ Root mean square
    """
    return pd.Series(index=src.index, data=np.sqrt(pd.Series(src*src).rolling(length, min_periods=1).sum().values/length))

# Smoothers

def SuperSmoother(src, period):
    """Ehlers SuperSmoother
    """
    a = exp(-RT2*PI/period)
    b = 2*a*cos(RT2*PI/period)
    c = a*a
    
    c2 = b
    c3 = -a*a
    c1 = 1 - c2 - c3

    s = np.pad(np.array(src), (period*2, 0), 'edge')
    s = signal.lfilter(b=[c1], a=[1, -c2, -c3], x=s)

    return pd.Series(index=src.index, data=s[period*2:])

def Decycler(src, cutoff):
    """ Ehlers decycler low pass filter
    """
    a1 = (cos(PI/cutoff)+sin(PI/cutoff)-1) / cos(PI/cutoff)
    
    s = np.pad(np.array(src), (cutoff*2, 0), 'edge')    
    s = signal.lfilter(b=[a1/2, a1/2], a=[1, -(1-a1)], x=s)

    return pd.Series(index=src.index, data=s[cutoff*2:])

def Highpass2Pole(src, cutoff):
    """ Ehlers 2-pole high pass filter
    """
    a = (cos(PI/RT2/cutoff) + sin(PI/RT2/cutoff) - 1) / cos(PI/RT2/cutoff)
    b = 1 - a/2
    
    s = np.pad(np.array(src), (cutoff*2, 0), 'edge')    
    s = signal.lfilter(b=[b*b, -2*b*b, b*b], a=[1, -2*(1-a), (1-a)*(1-a)], x=s)

    return pd.Series(index=src.index, data=s[cutoff*2:])

def DecyclerOscillator(src, cutoff1, cutoff2):
    """ Ehlers decycler oscillator
    Difference between 2 high pass filters with cutoff1 < cutoff2
    """
    return Highpass2Pole(src, cutoff2) - Highpass2Pole(src, cutoff1)

def Roof(src, cutoff_low=10, cutoff_high=48):
    """ Ehlers roofing filter
    """
    hp = Highpass2Pole(src, cutoff_high)
    return SuperSmoother(hp, cutoff_low)

def SuperPassband(src, fast=40, slow=60):
    """ Ehlers Super Passband Filter
    """
    a1 = 5/fast
    a2 = 5/slow

    s = np.pad(np.array(src), (slow*2, 0), 'edge')
    s = signal.lfilter(b=[(a1-a2), a2*(1-a1)-a1*(1-a2)], a=[1, -(2-a1-a2), (1-a1)*(1-a2)], x=s)
    return pd.Series(index=src.index, data=s[slow*2:])


def Bandpass(src, period=10, bandwidth=0.3):
    """ Ehlers bandpass filter
    """

    a2 = (cos(0.5*bandwidth*PI/period) + sin(0.5*bandwidth*PI/period) - 1)/cos(0.5*bandwidth*PI/period)
    hp = signal.lfilter(b=[(1+a2/2), -(1+a2/2)], a=[1, -(1-a2)], x=src)

    b1 = cos(2*PI/period)
    g1 = 1/cos(2*PI*bandwidth/period)
    a1 = g1 - math.sqrt(g1*g1-1)

    bp = signal.lfilter(b=[0.5*(1-a1), 0, -0.5*(1-a1)], a=[1, -b1*(1+a1), a1], x=hp)

    return pd.Series(index=src.index, data=bp)

def GaussianFilter(src, period, npoles):
    """ N-pole Gaussian filter (up to 4)
    """
    B = (1-math.cos(2*math.pi/period)) / math.pow(2, 1/npoles-1)
    A = -B + math.sqrt(math.pow(B,2)+2*B)
    
    _src = np.pad(np.array(src), (period*2, 0), 'edge')
    result = list(_src[:4])

    if npoles==1:
        for i in range(4, len(_src)):
            result.append(A*_src[i] + (1-A)*(result[i-1]))
    elif npoles==2:
        for i in range(4, len(_src)):
            result.append(math.pow(A, 2)*_src[i] + 2*(1-A)*(result[i-1]) - math.pow(1-A, 2)*result[i-2])
    elif npoles==3:
        for i in range(4, len(_src)):
            result.append(math.pow(A, 3)*_src[i] + 3*(1-A)*(result[i-1]) - 3*math.pow(1-A, 2)*result[i-2] + math.pow(1-A, 3)*result[i-3])
    elif npoles==4:
        for i in range(4, len(_src)):
            result.append(math.pow(A, 4)*_src[i] + 4*(1-A)*(result[i-1]) - 6*math.pow(1-A, 2)*result[i-2] + 4*math.pow(1-A, 3)*result[i-3] -math.pow(1-A, 4)*result[i-4])

    return pd.Series(index=src.index, data=result[period*2:])