import matplotlib.pyplot as plt


def plot(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y)

def plot2(x,y1,y2):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y1)
    ax.plot(x,y2)
