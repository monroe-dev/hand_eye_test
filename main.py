# import matplotlib as mpl
# mpl.use('agg')

import matplotlib.pyplot as plt
import simulate

numTest = 10
numPose = 10
numMethod = 4
# CALIB_HAND_EYE_DANIILIDIS = 4
# CALIB_HAND_EYE_ANDREFF = 3
# CALIB_HAND_EYE_HORAUD = 2
# CALIB_HAND_EYE_PARK = 1
# CALIB_HAND_EYE_TSAI = 0

# Test
rvec_diff, t_diff, mean_rvec_diff, mean_t_diff, std_rvec_diff, std_t_diff, max_rvec_diff, max_t_diff = simulate.validate(numTest, numPose, numMethod)

# Print
print('Error analyis')
for i in range(numMethod):
    print('Method', i, ': (Mean / Std / Max)')
    print('Orientation(Rotation vector): ', mean_rvec_diff[i], '/', std_rvec_diff[i], '/', max_rvec_diff[i])
    print('Position: ', mean_t_diff[i], '/', std_t_diff[i], '/', max_t_diff[i])

# Graph
plt1 = plt.subplot(221)
plt2 = plt.subplot(222)
plt3 = plt.subplot(223)
plt4 = plt.subplot(224)

for i in range(numMethod):
    plt1.plot(rvec_diff[:, i])
    plt2.plot(t_diff[:, i])
    plt1.set_xlabel('Measurement number')
    plt2.set_xlabel('Measurement number')

plt1.set_title('Orientation error')
plt2.set_title('Position error')
plt1.legend(('TSA', 'PAR', 'HOR', 'DAN', 'AND'))
plt2.legend(('TSA', 'PAR', 'HOR', 'DAN', 'AND'))

plt3.boxplot(rvec_diff)
plt3.set_xlabel('Method')
plt4.boxplot(t_diff)
plt4.set_xlabel('Method')

# plt.grid(True)
plt.show()
