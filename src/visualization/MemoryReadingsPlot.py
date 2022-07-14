import matplotlib.pyplot as plt
import seaborn as sns

from utils.viz_utils import GenericPlot


class MemoryReadingsPlot(GenericPlot):
    def __init__(self, cfg, dataframe):
        super().__init__(cfg, dataframe)

    def _plot(self):
        ax = sns.barplot(data=self.dataframe, x="mem_location", y="mean")
        ax.errorbar(
            data=self.dataframe, x="mem_location", y="mean", yerr="variance", ls=""
        )
        plt.suptitle("Memory readings")
