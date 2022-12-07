from matplotlib import pyplot as plt

loss = [0.2951, 0.1057, 0.0863, 0.0757, 0.0704, 0.0785, 0.0694, 0.0740, 0.0709, 0.0668, 0.0638, 0.0681, 0.0699, 0.0709, 0.0661]
plt.plot(range(len(loss)), loss, '-o')
plt.title("1 Hour training loss")
plt.show()

loss = [0.2621, 0.0862, 0.0801, 0.0740, 0.0718, 0.0739, 0.0660, 0.0703, 0.0689, 0.0777, 0.0656, 0.0751, 0.0664, 0.0664, 0.0631, 0.0683, 0.0635, 0.0651, 0.0721, 0.0631, 0.0697, 0.0646, 0.0684]
plt.plot(range(len(loss)), loss, '-o')
plt.title("10 Hour training loss")
plt.show()