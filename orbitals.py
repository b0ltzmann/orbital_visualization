#!/usr/bin/env python
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d
from scipy.special import sph_harm
from PIL import Image


class Orbital:
    def __init__(self, l, ml, view, colormap="RdBu", steps=30, points=200):
        self.l = l
        self.ml = ml
        self.view = view
        # Set the colormap:
        self.colormap = colormap
        # Number of steps in animation:
        self.steps = steps
        # Number of points for theta and phi:
        self.points = points
        self.theta, self.phi = np.meshgrid(
            np.linspace(0, np.pi, num=points), np.linspace(0, 2 * np.pi, num=points)
        )
        # Generate figure:
        self.fig = plt.figure()
        if ml == "all":
            self.fig.set_size_inches(12, 12)
            self.spec = gridspec.GridSpec(
                ncols=2 * self.l + 1, nrows=self.l + 1, figure=self.fig
            )
        elif abs(ml) > l:
            print("Choose an appropriate ml: -l ... +l")
            quit()
        else:
            self.fig.set_size_inches(4, 4)
            self.ax = self.fig.add_subplot(111, projection="3d")

    @property
    def factors(self):
        if self.view == "sphere":
            factors = [0]
        elif self.view == "orbital":
            factors = [1]
        elif self.view == "animation":
            factors = np.linspace(0, 1, num=self.steps)
        else:
            print("Choose an appropriate view: sphere, orbital, or animation")
            quit()
        return factors

    def spherical_harmonic(self, l, ml, theta, phi):
        """
        Calculates the Y_l^{ml} spherical harmonic given theta and phi.
        Linear combinations are used to generate real functions.
        Scipy (mathematics) convention is opposite of physics convention:
            theta   - azimuthal: 0 -> 2pi
            phi     - polar    : 0 -> pi
        To avoid confusion, theta and phi are swapped to the physics notation:
            phi     - azimuthal: 0 -> 2pi
            theta   - polar    : 0 -> pi
        input:
            l       (int)
            ml      (int)
            theta   (np array)
            phi     (np array)
        output:
            Y       (np array)
        """
        scipy_theta = phi
        scipy_phi = theta
        Y = sph_harm(abs(ml), l, scipy_theta, scipy_phi)
        Yn = sph_harm(-abs(ml), l, scipy_theta, scipy_phi)
        if ml < 0:
            Yreal = ((0 + 1j) / np.sqrt(2)) * (Yn - ((-1) ** ml) * Y)
        elif ml == 0:
            Yreal = Y
        elif ml > 0:
            Yreal = (1 / np.sqrt(2)) * (Yn + ((-1) ** ml) * Y)
        # Return real function (without complex (0j) terms):
        return Yreal.real

    def sphere_to_cart(self, r, theta, phi):
        """
        Converts from spherical coordinates to cartesian coordinates.
        input:
            r       (np array)
            theta   (np array)
            phi     (np array)
        output:
            x       (np array)
            y       (np array)
            z       (np array)
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def scale_rs(self, Y, factor):
        """
        Scale spherical harmonic between sphere (0) and orbital (1) by changing
        the magnitude of the r coordinates between 1 and the absolute value of
        the spherical harmonic.
        input:
            Y       (np array)
            factor  (float)
        output:
            rs      (np array)
        """
        Ynorm = Y / Y.max()
        rs = np.zeros((self.points, self.points))
        for i in range(len(rs[:, 0])):
            for j in range(len(rs[0, :])):
                val = abs(Ynorm[i, j])
                # Avoid dividing by zero:
                if val == 0:
                    r_scale = 1 - factor
                else:
                    r_scale = val / val - factor
                if r_scale <= val:
                    r_scale = val
                rs[i, j] = r_scale
        return rs

    def add_to_plot(self, ax, l, ml, factor):
        """
        Plots the l, ml spherical harmonic with a factor between sphere (0) and
        ortial (1).
        input:
            ax      (axis)
            l       (int)
            ml      (int)
            factor  (float)
        """
        # Remove the current plot:
        ax.cla()
        # Draw Cartesian axes and labels:
        ax.plot([-1, 1], [0, 0], [0, 0], color="k", lw=0.5)
        ax.plot([0, 0], [-1, 1], [0, 0], color="k", lw=0.5)
        ax.plot([0, 0], [0, 0], [-1, 1], color="k", lw=0.5)
        ax.text(1.1, 0, 0, "$x$", va="center")
        ax.text(0, -1.3, 0, "$y$", va="center")
        ax.text(0, 0, 1.1, "$z$", ha="center")
        # Set 3D view:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=10)
        ax.set_axis_off()
        # Add l, ml label:
        ax.text(0, 0, 1.3, "$l={0}$ \n $m_l={1}$".format(l, ml), ha="center")
        # Add spherical harmonic to the plot:
        cmap = cm.ScalarMappable(cmap=self.colormap)
        Y = self.spherical_harmonic(l, ml, self.theta, self.phi)
        rs = self.scale_rs(Y, factor)
        x, y, z = self.sphere_to_cart(rs, self.theta, self.phi)
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=cmap.to_rgba(Y),
            alpha=0.5,
            linewidth=0,
            rstride=2,
            cstride=2,
        )

    def save_frame(self):
        """
        Saves the current state of the figure as a frame for a gif.
        """
        buf = io.BytesIO()
        self.fig.savefig(buf, format="jpg", dpi=300, bbox_inches="tight")
        buf.seek(0)
        image = Image.open(buf)
        return image

    def savegif(self, frames):
        """
        Saves a list of a frames as a gif.
        input:
            frames      (list)
        """
        # Add the reverse order for loop:
        frames.extend(frames[::-1])
        firstframe = frames[0]
        firstframe.save(
            "./animation.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            fps=10,
            loop=0,
        )

    def savefig(self):
        """
        Saves the current state of the figure as a pdf.
        """
        self.fig.savefig("./plot.pdf", bbox_inches="tight")

    def generate_plot(self):
        """
        Main function to generate plot/animation.
        """
        if len(self.factors) == 1:
            if self.ml == "all":
                for l in range(self.l + 1):
                    for ml in range(-l, l + 1):
                        ax = self.fig.add_subplot(
                            self.spec[l, ml + self.l], projection="3d"
                        )
                        self.add_to_plot(ax, l, ml, self.factors[0])
                self.savefig()
            else:
                self.add_to_plot(self.ax, self.l, self.ml, self.factors[0])
                self.savefig()
        elif len(self.factors) > 1:
            frames = []
            if self.ml == "all":
                for factor in self.factors:
                    for l in range(self.l + 1):
                        for ml in range(-l, l + 1):
                            ax = self.fig.add_subplot(
                                self.spec[l, ml + self.l], projection="3d"
                            )
                            self.add_to_plot(ax, l, ml, factor)
                    frames.append(self.save_frame())
            else:
                for factor in self.factors:
                    self.add_to_plot(self.ax, self.l, self.ml, factor)
                    frames.append(self.save_frame())
            self.savegif(frames)


if __name__ == "__main__":
    in_view = str(input("Which view? (sphere, orbital, or animation)\n"))
    in_l = int(input("l value? (0, 1, 2 ... )\n"))
    try:
        in_ml = int(input("ml value? (-l ... +l -- or 'all')\n"))
    except:
        in_ml = "all"
    orbital = Orbital(in_l, in_ml, in_view)
    orbital.generate_plot()
