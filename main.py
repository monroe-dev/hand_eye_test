import matplotlib.pyplot as plt
import simulate

numTest = 2
numPose = 10
rvec_diff, t_diff = simulate.validate(numTest, numPose)

print(rvec_diff.shape)
# Graph
# plt.subplot(211)
# plt.plot(rvec_diff)
# plt.ylabel('Orientation error')
# plt.grid(True)
#
# plt.subplot(212)
# plt.plot(t_diff)
# plt.ylabel('Position error')
# plt.grid(True)
# plt.show()
