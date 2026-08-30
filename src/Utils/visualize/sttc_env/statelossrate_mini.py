from visualize_MANETAgents import DataVisualizer
import matplotlib.pyplot as plt

log_patterns = [
    "out/evaluations/final_v6/all/St_Tr/St_MA/3_1_*real_0/*ult_fin*",
    "out/evaluations/final_v6/all/St_Tr/St_MA/3_1_*real_0.5/*ult_fin*",
    "out/evaluations/final_v6/all/St_Tr/St_MA/3_1_*real_0.75/*ult_fin*",
    "out/evaluations/final_v6/all/St_Tr/St_MA/3_1_*real_0.875/*ult_fin*",
    "out/evaluations/final_v6/all/St_Tr/St_MA/3_1_*real_0.9375/*ult_fin*",
    "out/evaluations/final_v6/all/St_Tr/St_MA/3_1_*real_0.96875/*ult_fin*",
]
visualizor = DataVisualizer(*log_patterns)
visualizor.plotStepsMeanSmallRollingWindow(jammer_ep_step=500)
visualizor.storePlots("out/visualizations/evaluations/Sttc/TJ")
plt.show()
