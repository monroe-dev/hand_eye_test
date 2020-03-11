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
fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(111)

plt1 = fig.add_subplot(221)
plt2 = fig.add_subplot(222)
plt3 = fig.add_subplot(223)
plt4 = fig.add_subplot(224)

for i in range(numMethod):
    plt1.plot(rvec_diff[:, i])
    plt2.plot(t_diff[:, i])
    plt1.set_xlabel('Measurement number')
    plt2.set_xlabel('Measurement number')

plt1.grid(True)
plt2.grid(True)
plt3.grid(True)
plt4.grid(True)
plt1.set_title('Orientation error')
plt2.set_title('Position error')
plt1.legend(('TSA', 'PAR', 'HOR', 'DAN', 'AND'))
plt2.legend(('TSA', 'PAR', 'HOR', 'DAN', 'AND'))

plt3.boxplot(rvec_diff)
plt4.boxplot(t_diff)
plt3.set_xlabel('Method')
plt4.set_xlabel('Method')

# plt.grid(True)

fig.savefig('result_handeye_calibration.png', dpi=300)
plt.show()