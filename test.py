import matplotlib.pyplot as plt
import numpy as np
import time

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()


plt.show()

fig, ax = plt.subplots()



# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

yo = ax.barh(y_pos, performance, xerr=error, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.draw()

for rect, h in zip(yo, np.random.rand(len(people))):
	rect.set_height(h)

time.sleep(1)

fig.canvas.draw()



plt.draw()

#plt.show()