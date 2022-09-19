from map import Map, Obstacle
import numpy as np
from reference_path import ReferencePath
from spatial_bicycle_models import BicycleModel
import matplotlib.pyplot as plt
from MPC import MPC
from scipy import sparse
import json
import codecs
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +"/CubicSpline/")
try:
    import cubic_spline_planner
except:
    raise


def calc_yaw(x, y):
    if(len(x)!=len(y)):
        print('error in function: calc_yaw')
        return
    else:
        yaw = []
        for i in range(len(x)-1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            yaw.append(math.atan2(dy, dx))
        yaw.append(yaw[-1])
        return yaw
        
        
def get_switch_back_course(path, dl):
    ax = path[:,0]
    ay = path[:,1]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    return cx, cy, cyaw, ck
        
if __name__ == '__main__':
    
    # Load Dataset
    with open('trajactory1.json', 'r') as f1:
        traj1 = json.load(f1)
    with open('trajactory2.json', 'r') as f2:
        traj2 = json.load(f2)
    
    # Load Dataset
    path1x = traj1['state/past/x']
    path1y = traj1['state/past/y']
    path1k = (path1y[-1] - path1y[-2])/(path1x[-1] - path1x[-2])
    
    # Extend trajectory for testing.
    # You can delete this block when necessary
    for i in range(150):
        path1x.append(path1x[-1]+0.3)
        path1y.append(path1y[-1]+0.3*path1k)
    
    # Wrap x and y lists into path array(size: [n,2])
    path1 = np.zeros([len(path1x),2])
    for i in range(len(path1x)):
        path1[i][0] = path1x[i]
        path1[i][1] = path1y[i]
    
    # Load Dataset
    path2x = traj2['state/past/x']
    path2y = traj2['state/past/y']
    path2k = (path2y[-2] - path2y[-3])/(path2x[-2] - path2x[-3])
    
    # Extend trajectory for testing.
    # You can delete this block when necessary
    for i in range(150):
        path2x.append(path2x[-1]+0.3)
        path2y.append(path2y[-1]+0.3*path2k)
        
    # Wrap x and y lists into path array(size: [n,2])
    path2 = np.zeros([len(path2x),2])
    for i in range(len(path2x)):
        path2[i][0] = path2x[i]
        path2[i][1] = path2y[i]
    
    # Interpolate a new path by using cubic spline.
    # dl means the distance(meter) between each waypoint
    dl1=0.2
    dl2=0.3
    
    # FInally we can get x, y positions of the new interpolated path
    # cyaw means yaw angle list
    # ck means curvature list
    cx1, cy1, cyaw1, ck1 = get_switch_back_course(path1, dl1)
    cx2, cy2, cyaw2, ck2 = get_switch_back_course(path2, dl2)
    



    # Load map file
    map1 = Map(file_path='maps/sim_map.png', origin=[-50, -50],
              resolution=0.3)
    map2 = Map(file_path='maps/sim_map.png', origin=[-50, -50],
              resolution=0.3)
        
    # Specify waypoints
    wp_x1 = cx1
    wp_x2 = cx2
    wp_y1 = cy1
    wp_y2 = cy2

    # Specify path resolution
    path_resolution1 = 0.2  # m / wp
    path_resolution2 = 0.3  # m / wp

    # Create smoothed reference path
    reference_path1 = ReferencePath(map1, wp_x1, wp_y1, path_resolution1,
                                   smoothing_distance=5, max_width=30.0,
                                   circular=False)
    reference_path2 = ReferencePath(map2, wp_x2, wp_y2, path_resolution2,
                                   smoothing_distance=5, max_width=30.0,
                                   circular=False)

    # Instantiate motion model
    car1 = BicycleModel(length=3, width=2,
                       reference_path=reference_path1, Ts=0.05)
    car2 = BicycleModel(length=3, width=2,
                       reference_path=reference_path2, Ts=0.05)




    ##############
    # Controller #
    ##############

    N1 = 30
    N2 = 30
    Q1 = sparse.diags([1.0, 0.0, 0.0])
    Q2 = sparse.diags([1.0, 0.0, 0.0])
    R1 = sparse.diags([0.5, 0.0])
    R2 = sparse.diags([0.5, 0.0])
    QN1 = sparse.diags([1.0, 0.0, 0.0])
    QN2 = sparse.diags([1.0, 0.0, 0.0])

    v_max1 = 18.0  # m/s
    v_max2 = 56.0  # m/s
    delta_max1 = 0.66  # rad
    delta_max2 = 0.66  # rad
    ay_max1 = 15.0  # m/s^2
    ay_max2 = 15.0  # m/s^2
    InputConstraints1 = {'umin': np.array([0.0, -np.tan(delta_max1)/car1.length]),
                        'umax': np.array([v_max1, np.tan(delta_max1)/car1.length])}
    InputConstraints2 = {'umin': np.array([0.0, -np.tan(delta_max2)/car2.length]),
                        'umax': np.array([v_max2, np.tan(delta_max2)/car2.length])}
    StateConstraints1 = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                        'xmax': np.array([np.inf, np.inf, np.inf])}
    StateConstraints2 = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                        'xmax': np.array([np.inf, np.inf, np.inf])}
    mpc1 = MPC(car1, N1, Q1, R1, QN1, StateConstraints1, InputConstraints1, ay_max1)
    mpc2 = MPC(car2, N2, Q2, R2, QN2, StateConstraints2, InputConstraints2, ay_max2)

    # Compute speed profile
    a_min1 = -0.1  # m/s^2
    a_min2 = -0.1  # m/s^2
    a_max1 = 0.5  # m/s^2
    a_max2 = 0.5  # m/s^2
    SpeedProfileConstraints1 = {'a_min': a_min1, 'a_max': a_max1,
                               'v_min': 0.0, 'v_max': v_max1, 'ay_max': ay_max1}
    SpeedProfileConstraints2 = {'a_min': a_min2, 'a_max': a_max2,
                               'v_min': 0.0, 'v_max': v_max2, 'ay_max': ay_max2}
    car1.reference_path.compute_speed_profile(SpeedProfileConstraints1)
    car2.reference_path.compute_speed_profile(SpeedProfileConstraints2)

    ##############
    # Simulation #
    ##############

    # Set simulation time to zero
    t = 0.0

    # Logging containers
    x_log1 = [car1.temporal_state.x]
    x_log2 = [car2.temporal_state.x]
    y_log1 = [car1.temporal_state.y]
    y_log2 = [car2.temporal_state.y]
    v_log1 = [0.0]
    v_log2 = [0.0]

    # Until arrival at end of path
    car1_data = dict()
    car1_data['state/id'] = [1535178]
    car1_data['state/current/x'] = []
    car1_data['state/current/y'] = []
    car1_data['state/current/length'] = []
    car1_data['state/current/width'] = []
    car1_data['state/current/velocity_x'] = []
    car1_data['state/current/velocity_y'] = []
    car1_data['state/current/bbox_yaw'] = []
    car1_data['state/future/x'] = []
    car1_data['state/future/y'] = []
    car1_data['state/future/length'] = []
    car1_data['state/future/width'] = []
    car1_data['state/future/velocity_x'] = []
    car1_data['state/future/velocity_y'] = []
    car1_data['state/future/bbox_yaw'] = []
    
    car2_data = dict()
    car2_data['state/id'] = [1535205]
    car2_data['state/current/x'] = []
    car2_data['state/current/y'] = []
    car2_data['state/current/length'] = []
    car2_data['state/current/width'] = []
    car2_data['state/current/velocity_x'] = []
    car2_data['state/current/velocity_y'] = []
    car2_data['state/current/bbox_yaw'] = []
    car2_data['state/future/x'] = []
    car2_data['state/future/y'] = []
    car2_data['state/future/length'] = []
    car2_data['state/future/width'] = []
    car2_data['state/future/velocity_x'] = []
    car2_data['state/future/velocity_y'] = []
    car2_data['state/future/bbox_yaw'] = []
    
    while (car1.s < reference_path1.length and car2.s < reference_path2.length):
        plt.clf()
        plt.scatter(path1x, path1y, c='green', s=10)
        plt.scatter(path2x, path2y, c='green', s=10)
        
        # Update Costmaps
        obs1 = [Obstacle(cx=car2.temporal_state.x, cy=car2.temporal_state.y, radius=0.1)]
        obs2 = [Obstacle(cx=car1.temporal_state.x, cy=car1.temporal_state.y, radius=0.1)]
        map1.add_obstacles(obs1)
        map2.add_obstacles(obs2)
        car1.reference_path.map = map1
        car2.reference_path.map = map2
        
        distance = math.sqrt((car1.temporal_state.x - car2.temporal_state.x)**2 + (car1.temporal_state.y - car2.temporal_state.y)**2)
        
        # Get control signals
        u1 = mpc1.get_control()
        u2 = mpc2.get_control()
        #if(distance>=10):
        #    u2 = mpc2.get_control()

        # Simulate car
        car1.drive(u1)
        car2.drive(u2)
        #if(distance>=10):
        #    car2.drive(u2)

        # Log car state
        x_log1.append(car1.temporal_state.x)
        x_log2.append(car2.temporal_state.x)
        y_log1.append(car1.temporal_state.y)
        y_log2.append(car2.temporal_state.y)
        v_log1.append(u1[0])
        v_log2.append(u2[0])

        # Increment simulation time
        t += (car1.Ts + car2.Ts)/2

        # Plot path and drivable area
        reference_path1.show()
        reference_path2.show()

        # Plot car
        car1.show()
        car2.show()

        # Plot MPC prediction
        mpc1.show_prediction()
        mpc2.show_prediction()
        
        map1.clear_obstacles(obs1)
        map2.clear_obstacles(obs2)
        
        # Store data
        car1_data['state/current/x'].append(car1.temporal_state.x)
        car2_data['state/current/x'].append(car2.temporal_state.x)
        car1_data['state/current/length'].append(3)
        car2_data['state/current/length'].append(3)
        car1_data['state/current/width'].append(2)
        car2_data['state/current/width'].append(2)
        car1_data['state/current/y'].append(car1.temporal_state.y)
        car2_data['state/current/y'].append(car2.temporal_state.y)
        car1_data['state/current/velocity_x'].append(u1[0]*math.cos(car1.temporal_state.psi))
        car2_data['state/current/velocity_x'].append(u2[0]*math.cos(car2.temporal_state.psi))
        car1_data['state/current/velocity_y'].append(u1[0]*math.sin(car1.temporal_state.psi))
        car2_data['state/current/velocity_y'].append(u2[0]*math.sin(car2.temporal_state.psi))
        car1_data['state/future/x'].append(mpc1.current_prediction[0])
        car2_data['state/future/x'].append(mpc2.current_prediction[0])
        car1_data['state/future/y'].append(mpc1.current_prediction[1])
        car2_data['state/future/y'].append(mpc2.current_prediction[1])
        car1_data['state/current/bbox_yaw'].append(car1.temporal_state.psi)
        car2_data['state/current/bbox_yaw'].append(car2.temporal_state.psi)
        
        
        
        
        car1_future_yaw = calc_yaw(mpc1.current_prediction[0], mpc1.current_prediction[1])
        car2_future_yaw = calc_yaw(mpc2.current_prediction[0], mpc2.current_prediction[1])
        car1_data['state/future/bbox_yaw'].append(car1_future_yaw)
        car2_data['state/future/bbox_yaw'].append(car2_future_yaw)
    
    
    
    
        car1_future_vel_x = []
        for i in range(len(mpc1.current_prediction[0])):
            if(i==0):
                pass
            else:
                current_x = mpc1.current_prediction[0][i]
                last_x = mpc1.current_prediction[0][i]
                car1_future_vel_x.append((current_x-last_x)/car1.Ts)
        car1_future_vel_x.append(car1_future_vel_x[-1])
        car1_data['state/future/velocity_x'].append(car1_future_vel_x)

        car2_future_vel_x = []
        for i in range(len(mpc2.current_prediction[0])):
            if(i==0):
                pass
            else:
                current_x = mpc2.current_prediction[0][i]
                last_x = mpc2.current_prediction[0][i]
                car2_future_vel_x.append((current_x-last_x)/car2.Ts)
        car2_future_vel_x.append(car2_future_vel_x[-1])
        car2_data['state/future/velocity_x'].append(car2_future_vel_x)
        
        car1_future_vel_y = []
        for i in range(len(mpc1.current_prediction[1])):
            if(i==0):
                pass
            else:
                current_y = mpc1.current_prediction[1][i]
                last_y = mpc1.current_prediction[1][i]
                car1_future_vel_y.append((current_y-last_y)/car1.Ts)
        car1_future_vel_y.append(car1_future_vel_y[-1])
        car1_data['state/future/velocity_y'].append(car1_future_vel_y)
        
        car2_future_vel_y = []
        for i in range(len(mpc2.current_prediction[1])):
            if(i==0):
                pass
            else:
                current_y = mpc2.current_prediction[1][i]
                last_y = mpc2.current_prediction[1][i]
                car2_future_vel_y.append((current_y-last_y)/car2.Ts)
        car2_future_vel_y.append(car2_future_vel_y[-1])
        car2_data['state/future/velocity_y'].append(car2_future_vel_y)
        
        
        
        
        # Set figure title
        plt.title('MPC Simulation: v(t): {:.2f}, delta(t): {:.2f}, Duration: '
                  '{:.2f} s'.format(u1[0], u1[1], t))
        plt.axis('off')
        plt.pause(0.00001)
    
    
    
    
    with codecs.open('car1_mpc_output.json','a', 'utf-8') as outf:
        json.dump(car1_data, outf, ensure_ascii=False)
        outf.write('\n')
        
    with codecs.open('car2_mpc_output.json','a', 'utf-8') as outf:
        json.dump(car2_data, outf, ensure_ascii=False)
        outf.write('\n')
