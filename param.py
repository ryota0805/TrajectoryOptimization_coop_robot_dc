#パラメータ管理class
import numpy as np
import env

class Parameter:
    env_data = env.Env()
    N = 20                                                     #系列データの長さ
    M = 11                                                       #設計変数の種類の個数(thetaは冗長な変数)
    
    #初期状態と終端状態
    set_cons = {'initial_x1'     :True,                          #境界条件をセットするかどうか
                'terminal_x1'    :True, 
                'initial_y1'     :True, 
                'terminal_y1'    :True, 
                'initial_x2'     :True, 
                'terminal_x2'    :True, 
                'initial_y2'     :True, 
                'terminal_y2'    :True,
                'initial_theta1' :True,
                'terminal_theta1':True,
                'initial_theta2' :True, 
                'terminal_theta2':True,
                'initial_omega1' :False,
                'terminal_omega1':True,
                'initial_omega2' :False,
                'terminal_omega2':True,
                'initial_v1'     :False, 
                'terminal_v1'    :True,
                'initial_v2'     :False, 
                'terminal_v2'    :True,
                'initial_theta'  :False,
                'terminal_theta' :False
                }
    
    dt = 1                                                      #刻み幅[s]                                             
    d1 = 0
    d2 = 0.6+0.25*2
    
    #ロボットシステムの中心座標と姿勢からx1,y1,x2,y2の境界条件を計算する
    initial_x = 0
    terminal_x = 10
    
    initial_y = 0
    terminal_y = 4
    
    initial_theta = 0
    terminal_theta = 0
    
    initial_x1 = initial_x - d2/2*np.cos(initial_theta)                                             #x[m]
    terminal_x1 = terminal_x - d2/2*np.cos(terminal_theta)                                         #x[m]
    
    initial_y1 = initial_y - d2/2*np.sin(initial_theta)                                              #y[m]
    terminal_y1 = terminal_y - d2/2*np.sin(terminal_theta)                                              #y[m]
    
    initial_x2 = initial_x + d2/2*np.cos(initial_theta)                                             #x[m]
    terminal_x2 = terminal_x + d2/2*np.cos(terminal_theta)                                         #x[m]
    
    initial_y2 = initial_y + d2/2*np.sin(initial_theta)                                               #y[m]
    terminal_y2 = terminal_y + d2/2*np.sin(terminal_theta)                                              #y[m]

    initial_theta1 = 0                                         #theta[rad]
    terminal_theta1 = 0                                          #theta[rad]
    
    initial_theta2 = 0
    terminal_theta2 = 0
    
    initial_omega1 = 0
    terminal_omega1 = 0
    
    initial_omega2 = 0
    terminal_omega2 = 0
    
    initial_v1 = 0                                               #v[m/s]
    terminal_v1 = 0                                              #v[m/s]
    
    initial_v2 = 0                                               #v[m/s]
    terminal_v2 = 0                                              #v[m/s]
    
    #変数の範囲
    x_min = env_data.x_range[0]                                                  #x[m]
    x_max = env_data.x_range[1]                                                  #x[m]
    y_min = env_data.y_range[0]                                                 #y[m]
    y_max = env_data.y_range[1]                                                  #y[m]
    theta1_min = -np.pi * 180/ 180                                         #theta[rad]
    theta1_max = np.pi *180/180                                          #tehta[rad]
    theta2_min = -np.pi * 180/ 180                                           #theta[rad]
    theta2_max = np.pi * 180/ 180  
    omega1_min = -np.pi/6
    omega1_max = np.pi/6
    omega2_min = -np.pi/6
    omega2_max = np.pi/6
    v1_min = -2                                                   #v[m/s]
    v1_max = 2                                                   #v[m/s]
    v2_min = -2                                                   #v[m/s]
    v2_max = 2                                                   #v[m/s]
    
    theta_min = -np.pi * 180/ 180                                         #theta[rad]
    theta_max = np.pi *180/180   
    
    #ロボットの把持部分の最大値
    phi_max = np.pi/6
    WayPoint = np.array([[initial_x, initial_y],                #初期パス　[x, y] 
                        [3, 3],
                        [5, 3],
                        [terminal_x, terminal_y]])     

    #障害物のパラメータ 
    #　(x, y, r)
    #　x　: 円の中心座標
    #　y　: 円の中心座標
    #　r　: 半径 
    obstacle_list = [(10, -1, 3), (20, 1, 3)]                   #障害物のパラメータが格納されたリスト
    
    
    #wallのパラメータ
    wall_thick = 1                                #wallの厚さ
    margin = 2
    
    #robot size
    robot_size = 1
    
    #v, omegaの初期値との誤差
    error_omega = 0.001
    error_v = 0.001
    error_robot_d = 0.01