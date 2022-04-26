from enum import Enum
import matplotlib.pyplot as plt


class PlotType(Enum):
    scatter = "scatter"
    line = "line"
    errorbar = "errorbar"


def add_plot(cls):
    """Adds plotting to a ztf LC"""

    def plot(self, x: str, y: str, yerr=None, kind: PlotType = PlotType("scatter"), ax=None, fig=None, plot_kwargs={}):
        x = self[x]
        y = self[y]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 9))

        plot_func = ax.plot
        kwargs = {}
        match kind:
            case PlotType.line:
                plot_func = ax.plot
                kwargs = {
                    "linestyle": "-",
                    "marker": "None",
                }

            case PlotType.scatter:
                plot_func = ax.plot
                kwargs = {
                    "linestyle": "None",
                    "marker": "o",
                }

            case PlotType.errorbar:
                plot_func = ax.errorbar
                if yerr is None:
                    raise AttributeError("Yerr must be defined if PlotType is errorbar")
                yerr = self[yerr]

                kwargs = {
                    "linestyle": "None",
                    "marker": "o",
                    "yerr": yerr,
                }
        kwargs = kwargs | plot_kwargs
        plot_func(x, y, **kwargs)

        return fig, ax

    def plot_diff_flux(self, ax=None, marker="o", linestyle="None", xmin=None, xmax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 9))

        ax.plot(self["jd"], self["forcediffimflux"], marker=marker, linestyle=linestyle)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(which="major", rotation=45)

        # Update x axis limits:
        curr_xmin, curr_xmax = ax.get_xlim()
        curr_xmin = xmin if xmin is not None else curr_xmin
        curr_xmax = xmax if xmax is not None else curr_xmax
        ax.set_xlim(curr_xmin, curr_xmax)

        return ax

    setattr(cls, "plot", plot)
    setattr(cls, "plot_diff_flux", plot_diff_flux)
    return cls
