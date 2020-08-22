import pylab as pl

from matplotlib import animation

def images(ax, h, *args, **kwargs):

    # First set up the figure, the axis, and the plot element we want to animate
    im = ax.imshow(h[0], *args, **kwargs)

    # initialization function: plot the background of each frame
    def init():
        im.set_array([])
        return im,

    def animate(ii):
        im.set_array(h[ii])
        ax.set_title('%s'%ii)
        pl.draw()
        return im,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(ax.get_figure(), animate, frames=len(h), interval=50, blit=True)

    return anim

#anim.save('animation_image.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
