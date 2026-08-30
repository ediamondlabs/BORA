from visualize_MANETAgents import DataVisualizer
import matplotlib.pyplot as plt

log_patterns = [
    "out/evaluations/final_v6/all/*/St_MA/?_1_*_real_0.5/*ult_fin*",
    "out/evaluations/final_v6/all/*/St_MA/?_1_*_real_0.5/*Gredy*",
]
visualizor = DataVisualizer(*log_patterns)
visualizor.plotMeanStabilizedThroughputsOverNodesByAttacker(jammer_ep_steps=500)
visualizor.storePlots("out/visualizations/evaluations/Sttc")
plt.show()
