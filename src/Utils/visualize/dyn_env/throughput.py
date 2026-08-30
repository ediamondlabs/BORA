from visualize_MANETAgents import DataVisualizer
import matplotlib.pyplot as plt

log_patterns = [
    "out/evaluations/final_v6/all/St_Tr/Dy_MA/*_real_0.5/*ult_fin*",
    "out/evaluations/final_v6/all/St_Tr/Dy_MA/*_real_0.5/*Gredy*",
]
visualizor = DataVisualizer(*log_patterns)
visualizor.plotMeanThroughputs(500)
visualizor.storePlots("out/visualizations/evaluations/Dyn/TJ")
plt.show()
