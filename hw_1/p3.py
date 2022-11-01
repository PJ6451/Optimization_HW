import numpy
import matplotlib.pyplot as plt

x = numpy.linspace(-20,20,1000)
y = numpy.linspace(-20,20,1000)

X,Y = numpy.meshgrid(x,y)

Z = 5 - 5*X - 2*Y + 2*X*X + 5*X*Y + 6*Y*Y

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, [5,10,20,30,40,50,60,70,80,90,100,120,140,160])
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('HW 1, Problem 3')
plt.show()