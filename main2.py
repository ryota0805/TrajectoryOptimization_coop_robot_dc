from param import Parameter as p
import GenerateInitialPath
import util
import constraints_dc
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import objective_function 
import plot
import time
import animation


# 計測開始
start_time = time.time()

#WayPointから設計変数の初期値を計算する
#cubicX, cubicY = GenerateInitialPath.cubic_spline()
cubicX, cubicY = GenerateInitialPath.cubic_spline_by_waypoint(p.WayPoint)
x1, y1, x2, y2, theta1, theta2, omega1, omega2, v1, v2, theta = GenerateInitialPath.generate_initialpath2(cubicX, cubicY)
#xs1, ys1, xs2, ys2, thetas1, thetas2, omega1, omega2, v1, v2 = GenerateInitialPath.generate_initialpath_randomly(cubicX, cubicY)
xs1, ys1, xs2, ys2, thetas1, thetas2, omega1, omega2, v1, v2, thetas = GenerateInitialPath.initial_zero(0.1)
trajectory_matrix = np.array([x1, y1, x2, y2, theta1, theta2, omega1, omega2, v1, v2, theta])
trajectory_vector = util.matrix_to_vector(trajectory_matrix)

#目的関数の設定
func = objective_function.objective_function
jac_of_objective_function = objective_function.jac_of_objective_function

args = (1, 1)

#制約条件の設定
cons = constraints_dc.generate_cons_with_jac()

#変数の範囲の設定
bounds = constraints_dc.generate_bounds()

#オプションの設定
options = {'maxiter':10000, 'ftol': 1e-6}
print(len(cons))
# 例として、制約条件（constraints）のうち、インデックス0および24のジャグ配列がサイズ不一致であると仮定

# ジャグ配列のサイズを確認し、必要であれば調整
arr_0 = cons[0]['jac'](trajectory_vector, *cons[0]['args'])
arr_24 = cons[24]['jac'](trajectory_vector, *cons[24]['args'])
print(np.shape(arr_24))
#最適化を実行
result = optimize.minimize(func, trajectory_vector, args = args, method='SLSQP', jac = jac_of_objective_function, constraints=cons, bounds=bounds, options=options)
#result = optimize.minimize(func, trajectory_vector, args = args, method='SLSQP', constraints=cons, bounds=bounds, options=options)

# 計測終了
end_time = time.time()

#最適化結果の表示
print(result)
trajectory_matrix = result.x.reshape(p.M, p.N)
x1, y1, x2, y2, theta1, theta2, omega1, omega2, v1, v2, theta = trajectory_matrix[0], trajectory_matrix[1], trajectory_matrix[2], trajectory_matrix[3], trajectory_matrix[4], trajectory_matrix[5], trajectory_matrix[6], trajectory_matrix[7], trajectory_matrix[8], trajectory_matrix[9], trajectory_matrix[10]

x, y = (x1 + x2)/2, (y1 + y2)/2

phi1 = theta1 - theta
phi2 = -(theta2 - theta)

plot.vis_env()
plot.vis_path(x, y)
plot.compare_path(x1, y1, x2, y2)
plot.vis_history_theta(theta, range_flag = True)
plot.history_robot_theta(theta1, theta2, range_flag = True)
plot.history_robot_phi(phi1, phi2, range_flag = True)
plot.history_robot_omega(omega1, omega2, range_flag = True)
plot.history_robot_v(v1, v2, range_flag = True)


# 経過時間を計算
elapsed_time = end_time - start_time
print(f"実行時間: {elapsed_time}秒")

print((x1-x2)**2 + (y1-y2)**2 - p.d2**2)
animation.gen_robot_movie(x, y, theta, theta1, theta2, omega1, omega2, v1, v2, x1, y1, x2, y2, is_interpolation=True, vis_v=True)
