from visualize_MANETAgents import DataVisualizer
import matplotlib.pyplot as plt

log_patterns = [
    "out/evaluations/state_loss_rate/St_Tr/St_MA/3_1_*real_0/*ult_fin*",
    "out/evaluations/state_loss_rate/St_Tr/St_MA/3_1_*real_0.5/*ult_fin*",
    "out/evaluations/state_loss_rate/St_Tr/St_MA/3_1_*real_0.75/*ult_fin*",
    "out/evaluations/state_loss_rate/St_Tr/St_MA/3_1_*real_0.875/*ult_fin*",
    "out/evaluations/state_loss_rate/St_Tr/St_MA/3_1_*real_0.9375/*ult_fin*",
    "out/evaluations/state_loss_rate/St_Tr/St_MA/3_1_*real_0.96875/*ult_fin*",
    "out/evaluations/state_loss_rate/St_Tr/St_MA/3_1_*real_1/*ult_fin*",
    "out/evaluations/state_loss_rate/St_Tr/St_MA/3_1_*real_*/*Gredy_Trf*",
]
visualizor = DataVisualizer(*log_patterns)
visualizor.plotStateLossRollingWindow(interval=(0, 5000))
visualizor.storePlots("out/visualizations/evaluations/Sttc/TJ/longer_eval")
plt.show()
