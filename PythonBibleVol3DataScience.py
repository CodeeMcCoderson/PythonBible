import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def numPy():
    a = np.array([10, 20 ,30],[40, 50, 60])
    b = np.array([1, 77 ,23])
    c = np.full((3, 5, 4), 7)
    d = np.zeros((3, 3))
    e = np.ones((2, 3, 4, 2))
    f = np.empty((4,4))
    g = np.arange(10, 50, 5)
    h = np.linspace(0, 100, 11)

    np.savetext('myarray.csv', a)
    i = np.loadtext('myarray.csv')

def matPlotLib():
    #plotting mathematical functions
    xvalues = np.linspace(0, 20, 100)
    yvalues = np.sin(xvalues)
    plt.plot(xvalues, yvalues)
    plt.show()
    #plotting scatter plot
    numbers = 10 * np.random.random(100)
    plt.plot(numbers, 'bo')
    plt.show()
    #plotting multiple graphs
    a = np.linspace(0, 5, 200)
    b1 = 2 * a
    b2 = a ** 2
    b3 = np.log(a)
    plt.plot(a, b1)
    plt.plot(a, b2)
    plt.plot(a, b3)
    plt.show()
    #subplots
    c = np.linspace(0, 5, 200)
    d1 = np.sin(c)
    d2 = np.sqrt(c)
    plt.subplot(211) #2 rows, 1 column, index (1 being top)
    plt.plot(c, d1, 'r-')
    plt.subplot(212)
    plt.plot(c, d2, 'g--')
    plt.show()
    #labeling
    e = np.linspace(0, 50, 100)
    f = np.sin(e)
    plt.title('Sine Function')
    plt.suptitle('Data Science')
    plt.grid(True)
    plt.xlabel('x-values')
    plt.ylabel('y-values')
    plt.plot(e,f)
    #legends
    g = np.linspace(10, 50, 100)
    h1 = np.sin(g)
    h2 = np.cos(g)
    h3 = np.log(g/3)
    plt.plot(g, h1, 'b-', label = 'Sine')
    plt.plot(g, h2, 'r-', label = 'Cosine')
    plt.plot(g, h3, 'g-', label = 'Logaritm')
    plt.legend(loc='upper left')
    plt.show()
    #saving a diagram
    #plt.savefig('functions.png')
def matPlotLib2():
    #bar chart
    #pie chart
    #scatter plot
    m = np.random.rand(50)
    n = np.random.rand(50)
    plt.scatter(m, n)
    plt.show()
    # box plot
    # 3D plot
    ax = plt.axes(projection='3d')
    plt.show()
    z = np.linspace(0, 20, 100)
    q = np.sin(z)
    u = np.cos(z)
    ax = plt.axes(projection='3d')
    ax.plot3D(z, q, u)
    plt.show()
    #surface plot
    ax = plt.axes(projection='3d')
    def z_function(r, t):
        return np.sin(np.sqrt(r ** 2 + t ** 2))
    r = np.linspace(-5, 5, 50)
    t = np.linspace(-5, 5, 50)
    R, T = np.meshgrid(r, t)
    z = z_function(R, T)
    ax.plot_surface(R, T, z)
    plt.show()




matPlotLib2()
matPlotLib2()
