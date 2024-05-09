import math

def SLine(length=10, FStart=0, FStop=0, flexible=6, index=0):
    if index > length:
        index = length
    num = length / 2
    melo = flexible * (index - num) / num
    deno = 1.0 / (1 + math.exp(-melo))
    Fcurrent = FStart - (FStart - FStop) * deno
    return Fcurrent

# Example usage:
# len_val = 100
# FStart_val = 10.0
# FStop_val = 2.0
# flexible_val = 5.0
# index_val = 50
# result = motorPower_PowerSLine(len_val, FStart_val, FStop_val, flexible_val, index_val)
# result = motorPower_PowerSLine()
# print(f"Result: {result}")

# import matplotlib.pyplot as plt

# # Calculate S-curve results
# s_curve_results = [motorPower_PowerSLine(FStop=10, index=i) for i in range(11)]

# # Create scatter plot
# plt.scatter(range(11), s_curve_results, color='b', marker='o', label='S-curve points')
# plt.xlabel('Index')
# plt.ylabel('Fcurrent')
# plt.title('Scatter Plot of S-curve Results')
# plt.grid(True)
# plt.legend()
# plt.show()
