#不等式制約、等式制約を定義する
from param import Parameter as p
import util
import numpy as np
import env


#制約条件を計算するための運動学モデルの定式化
#dotx = v*cos(theta)
def kinematics_x(theta, v):
    return v*np.cos(theta)

#doty = v*cos(theta)
def kinematics_y(theta, v):
    return v*np.sin(theta)

#dot(theta) = omega
def kinematics_theta(omega):
    return omega

#xのcollocation point返す関数
def collocation_x(xs, thetas, vs):
    return 1/2*(xs[0] + xs[1]) + p.dt/8*(kinematics_x(thetas[0], vs[0]) - kinematics_x(thetas[1], vs[1]))

#yのcollocation point返す関数
def collocation_x(ys, thetas, vs):
    return 1/2*(ys[0] + ys[1]) + p.dt/8*(kinematics_y(thetas[0], vs[0]) - kinematics_y(thetas[1], vs[1]))

#thetaのcollocation point返す関数
def collocation_theta(thetas, omegas):
    return 1/2*(thetas[0] + thetas[1]) + p.dt/8*(kinematics_theta(omegas[0]) - kinematics_theta(omegas[1]))
    
    
def constraint(x, *args):
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    trajectory_matrix = x.reshape(p.M, p.N)
    x1, y1, x2, y2, theta1, theta2, omega1, omega2, v1, v2, theta = trajectory_matrix[0], trajectory_matrix[1], trajectory_matrix[2], trajectory_matrix[3], trajectory_matrix[4], trajectory_matrix[5], trajectory_matrix[6], trajectory_matrix[7], trajectory_matrix[8], trajectory_matrix[9], trajectory_matrix[10]
    
    if args[0] == 'model':
        i = args[1][1]
        if args[1][0] == 'x1':
            xs, thetas, omegas, vs = [x1[i], x1[i+1]], [theta1[i], theta1[i+1]], [omega1[i], omega1[i+1]], [v1[i], v1[i+1]]
            thetac = collocation_theta(thetas, omegas)
            vc = (vs[0] + vs[1])/2
            value = (xs[0] - xs[1]) + p.dt/6*(kinematics_x(thetas[0], vs[0]) + 4*kinematics_x(thetac, vc) + kinematics_x(thetas[1], vs[1])) 
        
        elif args[1][0] == 'y1':
            ys, thetas, omegas, vs = [y1[i], y1[i+1]], [theta1[i], theta1[i+1]], [omega1[i], omega1[i+1]], [v1[i], v1[i+1]]
            thetac = collocation_theta(thetas, omegas)
            vc = (vs[0] + vs[1])/2
            value = (ys[0] - ys[1]) + p.dt/6*(kinematics_y(thetas[0], vs[0]) + 4*kinematics_y(thetac, vc) + kinematics_y(thetas[1], vs[1])) 
            
        elif args[1][0] == 'theta1':
            thetas, omegas = [theta1[i], theta1[i+1]], [omega1[i], omega1[i+1]]
            omegac = (omegas[0] + omegas[1])/2
            value = (thetas[0] - thetas[1]) + p.dt/6*(kinematics_theta(omegas[0]) + 4*kinematics_theta(omegac) + kinematics_theta(omegas[1])) 
            
        elif args[1][0] == 'x2':
            xs, thetas, omegas, vs = [x2[i], x2[i+1]], [theta2[i], theta2[i+1]], [omega2[i], omega2[i+1]], [v2[i], v2[i+1]]
            thetac = collocation_theta(thetas, omegas)
            vc = (vs[0] + vs[1])/2
            value = (xs[0] - xs[1]) + p.dt/6*(kinematics_x(thetas[0], vs[0]) + 4*kinematics_x(thetac, vc) + kinematics_x(thetas[1], vs[1])) 
        
        elif args[1][0] == 'y2':
            ys, thetas, omegas, vs = [y2[i], y2[i+1]], [theta2[i], theta2[i+1]], [omega2[i], omega2[i+1]], [v2[i], v2[i+1]]
            thetac = collocation_theta(thetas, omegas)
            vc = (vs[0] + vs[1])/2
            value = (ys[0] - ys[1]) + p.dt/6*(kinematics_y(thetas[0], vs[0]) + 4*kinematics_y(thetac, vc) + kinematics_y(thetas[1], vs[1])) 
            
        elif args[1][0] == 'theta2':
            thetas, omegas = [theta2[i], theta2[i+1]], [omega2[i], omega2[i+1]]
            omegac = (omegas[0] + omegas[1])/2
            value = (thetas[0] - thetas[1]) + p.dt/6*(kinematics_theta(omegas[0]) + 4*kinematics_theta(omegac) + kinematics_theta(omegas[1])) 
             
        else:
            return 'Error'

        return value


    elif args[0] == 'avoid_obstacle':
        x, y = (x1 + x2)/2, (y1 + y2)/2
        if args[1][0] == 'rectangle':
            k, i = args[1][1], args[1][2]
            
            value = (((2*0.8/obs_rectangle[k][2]) ** 10) * (x[i] - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 10 + ((2*0.8/obs_rectangle[k][3]) ** 10) * (y[i] - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 10) - 1
            
            return value
        
        elif args[1][0] == 'circle':
            k, i = args[1][1], args[1][2]
            
            value = ((x[i] - obs_circle[k][0]) ** 2 + (y[i] - obs_circle[k][1]) ** 2) - (obs_circle[k][2] + p.robot_size) ** 2

            return value 
    
    elif args[0] == 'boundary':
        variable, ini_ter = args[1][0], args[1][1]
        
        if variable == 'x1':
            if ini_ter == 'ini':
                value = x1[0] - p.initial_x1
            
            elif ini_ter == 'ter':
                value = x1[-1] - p.terminal_x1
                
        elif variable == 'y1':
            if ini_ter == 'ini':
                value = y1[0] - p.initial_y1
            
            elif ini_ter == 'ter':
                value = y1[-1] - p.terminal_y1
                
        elif variable == 'x2':
            if ini_ter == 'ini':
                value = x2[0] - p.initial_x2
            
            elif ini_ter == 'ter':
                value = x2[-1] - p.terminal_x2
                
        elif variable == 'y2':
            if ini_ter == 'ini':
                value = y2[0] - p.initial_y2
            
            elif ini_ter == 'ter':
                value = y2[-1] - p.terminal_y2
                
        elif variable == 'theta1':
            if ini_ter == 'ini':
                value = theta1[0] - p.initial_theta1
            
            elif ini_ter == 'ter':
                value = theta1[-1] - p.terminal_theta1
                
        elif variable == 'theta2':
            if ini_ter == 'ini':
                value = theta2[0] - p.initial_theta2
            
            elif ini_ter == 'ter':
                value = theta2[-1] - p.terminal_theta2
                
        elif variable == 'omega1':
            if ini_ter == 'ini':
                value = omega1[0] - p.initial_omega1
            
            elif ini_ter == 'ter':
                value = omega1[-1] - p.terminal_omega1
                
        elif variable == 'omega2':
            if ini_ter == 'ini':
                value = omega2[0] - p.initial_omega2
            
            elif ini_ter == 'ter':
                value = omega2[-1] - p.terminal_omega2
                
        elif variable == 'v1':
            if ini_ter == 'ini':
                value = v1[0] - p.initial_v1
            
            elif ini_ter == 'ter':
                value = v1[-1] - p.terminal_v1 
                
        elif variable == 'v2':
            if ini_ter == 'ini':
                value = v2[0] - p.initial_v2
            
            elif ini_ter == 'ter':
                value = v2[-1] - p.terminal_v2
    
        return value
    
    
    #把持部分の角度の制約(面倒なのでロボットの角度の差がステアリング角の二倍より小さいという制約にしている)
    elif args[0] == 'steer':
        variable, i = args[1][0], args[1][1]
        if variable == 'theta1':
            value = (p.phi_max)**2 - (theta[i] - theta1[i])**2
            
        elif variable == 'theta2':
            value = (p.phi_max)**2 - (theta[i] - theta2[i])**2
        return value
    
    
    #不等式制約として与えるomega,vの初期値
    elif args[0] == 'ini':
        variable, i = args[1][0], args[1][1]
        if variable == 'omega1':
            value = p.error_omega**2 - (omega1[0] - p.initial_omega1)**2
            
        elif variable == 'omega2':
            value = p.error_omega**2 - (omega2[0] - p.initial_omega2)**2
            
        elif variable == 'v1':
            value = p.error_v**2 - (v1[0] - p.initial_v1)**2
            
        elif variable == 'v2':
            value = p.error_v**2 - (v2[0] - p.initial_v2)**2
            
        return value
    
    
    #2台のロボット間の距離がd2を維持するための不等式制約(等式制約だと解が見つからない)
    elif args[0] == 'robot_d':
        variable, i = args[1][0], args[1][1]
        
        value = p.error_robot_d**2 - (((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2 - p.d2**2))**2
        
        return value
    
    #2台のロボットと台車の姿勢を固定するための等式制約
    elif args[0] == 'posture':
        variable, i = args[1][0], args[1][1]
        
        value = np.arctan2(y2[i]-y1[i], x2[i]-x1[i]) - theta[i]
        
        return value
    
def jac_of_constraint(x, *args):
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    trajectory_matrix = x.reshape(p.M, p.N)
    x1, y1, x2, y2, theta1, theta2, omega1, omega2, v1, v2, theta = trajectory_matrix[0], trajectory_matrix[1], trajectory_matrix[2], trajectory_matrix[3], trajectory_matrix[4], trajectory_matrix[5], trajectory_matrix[6], trajectory_matrix[7], trajectory_matrix[8], trajectory_matrix[9], trajectory_matrix[10]
    
    jac_cons = np.zeros((p.M, p.N))
    
    if args[0] == 'model':
        i = args[1][1]
        if args[1][0] == 'x1':
            xs, thetas, omegas, vs = [x1[i], x1[i+1]], [theta1[i], theta1[i+1]], [omega1[i], omega1[i+1]], [v1[i], v1[i+1]]
            thetac = collocation_theta(thetas, omegas)
            vc = (vs[0] + vs[1])/2
            
            #x1[i]
            jac_cons[0, i] = 1
            #x1[i+1]
            jac_cons[0, i+1] = -1
            
            #theta1[i]
            jac_cons[4, i] = p.dt/6*(-v1[i]*np.sin(theta1[i]) + 4*(-vc)*np.sin(thetac)*1/2)
            #theta1[i+1]
            jac_cons[4, i+1] = p.dt/6*(-v1[i+1]*np.sin(theta1[i+1]) + 4*(-vc)*np.sin(thetac)*1/2)
            
            #omega1[i]
            jac_cons[6, i] = p.dt/6*(4*(-vc)*np.sin(thetac)*p.dt/8)
            #omega1[i+1]
            jac_cons[6, i+1] = p.dt/6*(4*(-vc)*np.sin(thetac)*(-p.dt/8))
            
            #v1[i]
            jac_cons[8, i] = p.dt/6*(np.cos(theta1[i]) + 4*(1/2*np.cos(thetac)))
            #v1[i+1]
            jac_cons[8, i+1] = p.dt/6*(np.cos(theta1[i+1]) + 4*(1/2*np.cos(thetac)))
        
        elif args[1][0] == 'y1':
            ys, thetas, omegas, vs = [y1[i], y1[i+1]], [theta1[i], theta1[i+1]], [omega1[i], omega1[i+1]], [v1[i], v1[i+1]]
            thetac = collocation_theta(thetas, omegas)
            vc = (vs[0] + vs[1])/2
            
            #y1[i]
            jac_cons[1, i] = 1
            #y1[i+1]
            jac_cons[1, i+1] = -1
            
            #theta1[i]
            jac_cons[4, i] = p.dt/6*(v1[i]*np.cos(theta1[i]) + 4*(vc)*np.cos(thetac)*1/2)
            #theta1[i+1]
            jac_cons[4, i+1] = p.dt/6*(v1[i+1]*np.cos(theta1[i+1]) + 4*(vc)*np.cos(thetac)*1/2)
            
            #omega1[i]
            jac_cons[6, i] = p.dt/6*(4*(vc)*np.cos(thetac)*p.dt/8)
            #omega1[i+1]
            jac_cons[6, i+1] = p.dt/6*(4*(vc)*np.cos(thetac)*(-p.dt/8))
            
            #v1[i]
            jac_cons[8, i] = p.dt/6*(np.sin(theta1[i]) + 4*(1/2*np.sin(thetac)))
            #v1[i+1]
            jac_cons[8, i+1] = p.dt/6*(np.sin(theta1[i+1]) + 4*(1/2*np.sin(thetac)))
            
        elif args[1][0] == 'theta1':
            thetas, omegas = [theta1[i], theta1[i+1]], [omega1[i], omega1[i+1]]
            omegac = (omegas[0] + omegas[1])/2
            
            #theta1[i]
            jac_cons[4, i] = 1
            #theta1[i+1]
            jac_cons[4, i+1] = -1
            
            #omega1[i]
            jac_cons[6, i] = p.dt/6*(3)
            #omega1[i+1] 
            jac_cons[6, i+1] = p.dt/6*(3)
            
        elif args[1][0] == 'x2':
            xs, thetas, omegas, vs = [x2[i], x2[i+1]], [theta2[i], theta2[i+1]], [omega2[i], omega2[i+1]], [v2[i], v2[i+1]]
            thetac = collocation_theta(thetas, omegas)
            vc = (vs[0] + vs[1])/2
            
            #x2[i]
            jac_cons[2, i] = 1
            #x2[i+1]
            jac_cons[2, i+1] = -1
            
            #theta2[i]
            jac_cons[5, i] = p.dt/6*(-v2[i]*np.sin(theta2[i]) + 4*(-vc)*np.sin(thetac)*1/2)
            #theta2[i+1]
            jac_cons[5, i+1] = p.dt/6*(-v2[i+1]*np.sin(theta2[i+1]) + 4*(-vc)*np.sin(thetac)*1/2)
            
            #omega2[i]
            jac_cons[7, i] = p.dt/6*(4*(-vc)*np.sin(thetac)*p.dt/8)
            #omega2[i+1]
            jac_cons[7, i+1] = p.dt/6*(4*(-vc)*np.sin(thetac)*(-p.dt/8))
            
            #v2[i]
            jac_cons[9, i] = p.dt/6*(np.cos(theta2[i]) + 4*(1/2*np.cos(thetac)))
            #v2[i+1]
            jac_cons[9, i+1] = p.dt/6*(np.cos(theta2[i+1]) + 4*(1/2*np.cos(thetac)))
        
        elif args[1][0] == 'y2':
            ys, thetas, omegas, vs = [y2[i], y2[i+1]], [theta2[i], theta2[i+1]], [omega2[i], omega2[i+1]], [v2[i], v2[i+1]]
            thetac = collocation_theta(thetas, omegas)
            vc = (vs[0] + vs[1])/2
            
            #y2[i]
            jac_cons[3, i] = 1
            #y2[i+1]
            jac_cons[3, i+1] = -1
            
            #theta2[i]
            jac_cons[5, i] = p.dt/6*(v2[i]*np.cos(theta2[i]) + 4*(vc)*np.cos(thetac)*1/2)
            #theta2[i+1]
            jac_cons[5, i+1] = p.dt/6*(v2[i+1]*np.cos(theta2[i+1]) + 4*(vc)*np.cos(thetac)*1/2)
            
            #omega2[i]
            jac_cons[7, i] = p.dt/6*(4*(vc)*np.cos(thetac)*p.dt/8)
            #omega2[i+1]
            jac_cons[7, i+1] = p.dt/6*(4*(vc)*np.cos(thetac)*(-p.dt/8))
            
            #v2[i]
            jac_cons[9, i] = p.dt/6*(np.sin(theta2[i]) + 4*(1/2*np.sin(thetac)))
            #v2[i+1]
            jac_cons[9, i+1] = p.dt/6*(np.sin(theta2[i+1]) + 4*(1/2*np.sin(thetac)))
            
        elif args[1][0] == 'theta2':
            thetas, omegas = [theta2[i], theta2[i+1]], [omega2[i], omega2[i+1]]
            omegac = (omegas[0] + omegas[1])/2
            
            #theta2[i]
            jac_cons[5, i] = 1
            #theta2[i+1]
            jac_cons[5, i+1] = -1
            
            #omega2[i]
            jac_cons[7, i] = p.dt/6*(3)
            #omega2[i+1] 
            jac_cons[7, i+1] = p.dt/6*(3)
            
        else:
            return 'Error'
        
        #ベクトルに直す
        jac_cons = jac_cons.flatten()
        
        return jac_cons
        
    #2台のロボットの中心座標に対して衝突判定を行う
    elif args[0] == 'avoid_obstacle':
        if args[1][0] == 'rectangle':
            k, i = args[1][1], args[1][2]
            
            #x1
            jac_cons[0, i] = (1/2)*10 * ((2*0.8/obs_rectangle[k][2]) ** 10) * ((x1[i] + x2[i])/2 - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 9
            #y1
            jac_cons[1, i] = (1/2)*10 * ((2*0.8/obs_rectangle[k][3]) ** 10) * ((y1[i] + y2[i])/2 - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 9
            #x2
            jac_cons[2, i] = (1/2)*10 * ((2*0.8/obs_rectangle[k][2]) ** 10) * ((x1[i] + x2[i])/2 - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 9
            #y2
            jac_cons[3, i] = (1/2)*10 * ((2*0.8/obs_rectangle[k][3]) ** 10) * ((y1[i] + y2[i])/2 - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 9
            
            #ベクトルに直す
            jac_cons = jac_cons.flatten()
        
            return jac_cons
            
        elif args[1][0] == 'circle':
            k, i = args[1][1], args[1][2]
            
            #x1
            jac_cons[0, i] = (1/2)*2 * ((x1[i] + x2[i])/2 - obs_circle[k][0])
            #y1
            jac_cons[1, i] = (1/2)*2 * ((y1[i] + y2[i])/2 - obs_circle[k][1])
            #x2
            jac_cons[0, i] = (1/2)*2 * ((x1[i] + x2[i])/2 - obs_circle[k][0])
            #y2
            jac_cons[1, i] = (1/2)*2 * ((y1[i] + y2[i])/2 - obs_circle[k][1])
            
            #ベクトルに直す
            jac_cons = jac_cons.flatten()
        
            return jac_cons
    
    
    elif args[0] == 'boundary':
        variable, ini_ter = args[1][0], args[1][1]
        
        if variable == 'x1':
            if ini_ter == 'ini':
                jac_cons[0, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[0, -1] = 1
                
        elif variable == 'y1':
            if ini_ter == 'ini':
                jac_cons[1, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[1, -1] = 1  
                
        elif variable == 'x2':
            if ini_ter == 'ini':
                jac_cons[2, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[2, -1] = 1
                
        elif variable == 'y2':
            if ini_ter == 'ini':
                jac_cons[3, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[3, -1] = 1  
                
        elif variable == 'theta1':
            if ini_ter == 'ini':
                jac_cons[4, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[4, -1] = 1  
                
        elif variable == 'theta2':
            if ini_ter == 'ini':
                jac_cons[5, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[5, -1] = 1  
                
        elif variable == 'omega1':
            if ini_ter == 'ini':
                jac_cons[6, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[6, -1] = 1
                
        elif variable == 'omega2':
            if ini_ter == 'ini':
                jac_cons[7, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[7, -1] = 1
                
        elif variable == 'v1':
            if ini_ter == 'ini':
                jac_cons[8, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[8, -1] = 1    
                
        elif variable == 'v2':
            if ini_ter == 'ini':
                jac_cons[9, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[9, -1] = 1    
                
        elif variable == 'theta':
            if ini_ter == 'ini':
                jac_cons[10, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[10, -1] = 1  
        
        #ベクトルに直す
        jac_cons = jac_cons.flatten()
    
        return jac_cons
    
    #把持部分の角度の制約
    elif args[0] == 'steer':
        variable, i = args[1][0], args[1][1]
        if variable == 'theta1':
            #theta1[i]
            jac_cons[4, i] = 2*(theta[i] - theta1[i])
            #theta[i]
            jac_cons[10, i] = -2*(theta[i] - theta1[i])
            
        elif variable == 'theta2':
            #theta2[i]
            jac_cons[5, i] =  2*(theta[i] - theta2[i])
            #theta[i]
            jac_cons[10, i] = -2*(theta[i] - theta2[i])
    
        #ベクトルに直す
        jac_cons = jac_cons.flatten()
    
        return jac_cons
    
    #不等式制約として与えるomega,vの初期値
    #把持部分の角度の制約
    elif args[0] == 'ini':
        variable, i = args[1][0], args[1][1]
        if variable == 'omega1':
            jac_cons[6, 0] =  -2*(omega1[i] - p.initial_omega1)
            
        elif variable == 'omega2':
            jac_cons[7, 0] =  -2*(omega2[i] - p.initial_omega2)
            
        elif variable == 'v1':
            jac_cons[8, 0] =  -2*(v1[i] - p.initial_v1)
            
        elif variable == 'v2':
            jac_cons[9, 0] =  -2*(v2[i] - p.initial_v2)
    
        #ベクトルに直す
        jac_cons = jac_cons.flatten()
    
        return jac_cons
    
    #2台のロボット間の距離がd2を維持するための不等式制約(等式制約だと解が見つからない)
    elif args[0] == 'robot_d':
        variable, i = args[1][0], args[1][1]
        #p.error_robot_d**2 - (((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2 - p.d2**2))**2
        
        #x1
        jac_cons[0, i] =  - 4*(((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2 - p.d2**2))*(x1[i] - x2[i])
        #y1
        jac_cons[1, i] =  - 4*(((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2 - p.d2**2))*(y1[i] - y2[i])
        #x2
        jac_cons[2, i] =  4*(((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2 - p.d2**2))*(x1[i] - x2[i])
        #y2
        jac_cons[3, i] =  4*(((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2 - p.d2**2))*(y1[i] - y2[i])
        
        #ベクトルに直す
        jac_cons = jac_cons.flatten()
        
        return jac_cons
    
    #2台のロボットと台車の姿勢を固定するための等式制約(ロボットの位置と台車の角度の内積)
    elif args[0] == 'posture':
        variable, i = args[1][0], args[1][1]
        
        #x1[i]
        jac_cons[0, i] = 1/(1 + ((y2[i]-y1[i])/(x2[i]-x1[i]))**2)*(y2[i]-y1[i])/(x2[i]-x1[i])**2
        #y1[i]
        jac_cons[1, i] = 1/(1 + ((y2[i]-y1[i])/(x2[i]-x1[i]))**2)*(-1)/(x2[i]-x1[i])
        #x2[i]
        jac_cons[2, i] = -1/(1 + ((y2[i]-y1[i])/(x2[i]-x1[i]))**2)*(y2[i]-y1[i])/(x2[i]-x1[i])**2
        #y2[i]
        jac_cons[3, i] = 1/(1 + ((y2[i]-y1[i])/(x2[i]-x1[i]))**2)*(1)/(x2[i]-x1[i])
        #theta[i]
        jac_cons[10, i] = -1
    
        #ベクトルに直す
        jac_cons = jac_cons.flatten()
        
        return jac_cons
    
def generate_cons_with_jac():
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    cons = ()
    
    #障害物回避のための不等式制約を追加する
    #矩形
    for k in range(len(obs_rectangle)):
        for i in range(p.N):
            args = ['avoid_obstacle', ['rectangle', k, i]]
            cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
            
            
    #円形
    for k in range(len(obs_circle)):
        for i in range(p.N):
            args = ['avoid_obstacle', ['circle', k, i]]
            cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
            
            
    #運動学モデルの制約からなる等式制約を追加する
    #x1
    for i in range(p.N-1):
        args = ['model', ['x1', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y1
    for i in range(p.N-1):
        args = ['model', ['y1', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    #theta1
    for i in range(p.N-1):
        args = ['model', ['theta1', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #x2
    for i in range(p.N-1):
        args = ['model', ['x2', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y2
    for i in range(p.N-1):
        args = ['model', ['y2', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    #theta2
    for i in range(p.N-1):
        args = ['model', ['theta2', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
        
    #境界値条件の等式制約を追加
    #x1初期条件
    if p.set_cons['initial_x1'] == False:
        pass
    else:
        args = ['boundary', ['x1', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #x1終端条件
    if p.set_cons['terminal_x1'] == False:
        pass
    else:
        args = ['boundary', ['x1', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)

    #y1初期条件
    if p.set_cons['initial_y1'] == False:
        pass
    else:
        args = ['boundary', ['y1', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y1終端条件
    if p.set_cons['terminal_y1'] == False:
        pass
    else:
        args = ['boundary', ['y1', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #x2初期条件
    if p.set_cons['initial_x2'] == False:
        pass
    else:
        args = ['boundary', ['x2', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #x2終端条件
    if p.set_cons['terminal_x2'] == False:
        pass
    else:
        args = ['boundary', ['x2', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y2初期条件
    if p.set_cons['initial_y2'] == False:
        pass
    else:
        args = ['boundary', ['y2', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y2終端条件
    if p.set_cons['terminal_y2'] == False:
        pass
    else:
        args = ['boundary', ['y2', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta1初期条件
    if p.set_cons['initial_theta1'] == False:
        pass
    else:
        args = ['boundary', ['theta1', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta1終端条件
    if p.set_cons['terminal_theta1'] == False:
        pass
    else:
        args = ['boundary', ['theta1', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta2初期条件
    if p.set_cons['initial_theta2'] == False:
        pass
    else:
        args = ['boundary', ['theta2', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta2終端条件
    if p.set_cons['terminal_theta2'] == False:
        pass
    else:
        args = ['boundary', ['theta2', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #omega1初期条件
    if p.set_cons['initial_omega1'] == False:
        pass
    else:
        args = ['boundary', ['omega1', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #omega1終端条件
    if p.set_cons['terminal_omega1'] == False:
        pass
    else:
        args = ['boundary', ['omega1', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #omega2初期条件
    if p.set_cons['initial_omega2'] == False:
        pass
    else:
        args = ['boundary', ['omega2', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #omega2終端条件
    if p.set_cons['terminal_omega2'] == False:
        pass
    else:
        args = ['boundary', ['omega2', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v1初期条件
    if p.set_cons['initial_v1'] == False:
        pass
    else:
        args = ['boundary', ['v1', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v1終端条件
    if p.set_cons['terminal_v1'] == False:
        pass
    else:
        args = ['boundary', ['v1', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v2初期条件
    if p.set_cons['initial_v2'] == False:
        pass
    else:
        args = ['boundary', ['v2', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v2終端条件
    if p.set_cons['terminal_v2'] == False:
        pass
    else:
        args = ['boundary', ['v2', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta初期条件
    if p.set_cons['initial_theta'] == False:
        pass
    else:
        args = ['boundary', ['theta', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta終端条件
    if p.set_cons['terminal_theta'] == False:
        pass
    else:
        args = ['boundary', ['theta', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    
    
    #ステアリング角の制約を加える
    for i in range(p.N):
        args = ['steer', ['theta1', i]]
        cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
        args = ['steer', ['theta2', i]]
        cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    
    #不等式制約としてomega,vの初期値を与える
    args = ['ini', ['omega1', 0]]
    cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    args = ['ini', ['omega2', 0]]
    cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)    
    
    args = ['ini', ['v1', 0]]
    cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    args = ['ini', ['v2', 0]]
    cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)    
    
    """
    #ロボット間の距離の制約
    for i in range(1, p.N-1):
        args = ['robot_d', [None, i]]
        cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    """
    #ロボットと台車の姿勢の等式制約
    for i in range(1, p.N-1):
        args = ['posture', [None, i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    return cons
    


#変数の数だけタプルのリストとして返す関数
def generate_bounds():
    
    #boundsのリストを生成
    bounds = []
    
    #x1の範囲
    for i in range(p.N):
        bounds.append((p.x_min, p.x_max))
        
    #y1の範囲
    for i in range(p.N):
        bounds.append((p.y_min, p.y_max))
        
    #x2の範囲
    for i in range(p.N):
        bounds.append((p.x_min, p.x_max))
        
    #y2の範囲
    for i in range(p.N):
        bounds.append((p.y_min, p.y_max))
        
    #theta1の範囲
    for i in range(p.N):
        bounds.append((p.theta1_min, p.theta1_max))
        
    #theta2の範囲
    for i in range(p.N):
        bounds.append((p.theta2_min, p.theta2_max))
        
    #omega1の範囲
    for i in range(p.N):
        bounds.append((p.omega1_min, p.omega1_max))
        
    #omega2の範囲
    for i in range(p.N):
        bounds.append((p.omega2_min, p.omega2_max))
        
    #v1の範囲
    for i in range(p.N):
        bounds.append((p.v1_min, p.v1_max))
        
    #v2の範囲
    for i in range(p.N):
        bounds.append((p.v2_min, p.v2_max))
        
    #thetaの範囲
    for i in range(p.N):
        bounds.append((p.theta_min, p.theta_max))
    
    return bounds