from visualize_MANETAgents import DataVisualizer
import matplotlib.pyplot as plt

log_patterns = [
    "out/evaluations/final_v6/all/St_Tr/St_MA/*_real_0.5/*ult_fin*",
    "out/evaluations/final_v6/all/St_Tr/St_MA/*_real_0.5/*Gredy*",
]
visualizor = DataVisualizer(*log_patterns)
visualizor.plotMeanStabilizedThroughputs(500)
visualizor.storePlots("out/visualizations/evaluations/Sttc/TJ")
plt.show()
