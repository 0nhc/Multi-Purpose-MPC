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

def calc_v(distance):
    obstacle_distance_range = 10 #meter
    if(distance<=obstacle_distance_range):
        return (obstacle_distance_range - 0.7*distance)
    else:
        return 0

if __name__ == '__main__':    
    # 加载地图文件
    map_data = Map(file_path='maps/sim_map.png', origin=[-50, -50], resolution=0.1)
    
    wp_x1 = [0, 10, 20, 30, 40, 50]
    wp_y1 = [0, 15, 15, 25, 45, 50]

    wp_x2 = [50, 40, 30, 20, 10, 0]
    wp_y2 = [0, 5, 25, 35, 35, 50]

    wp_x3 = [-10, 0, 10, 20, 30, 40, 50, 60]
    wp_y3 = [-15, 0, 15, 15, 25, 35, 40, 65]

    wp_x4 = [60, 50, 40, 30, 20, 10, 0, -10]
    wp_y4 = [-5, 0, 5, 25, 35, 35, 55, 55]

    x_data = [wp_x1, wp_x2, wp_x4]
    y_data = [wp_y1, wp_y2, wp_y4]
    num_paths = len(x_data)
    
    # 每一时刻的障碍物坐标记录
    obs_cache = []

    # 遍历所有轨迹
    for i in range(num_paths):
        wp_x = x_data[i]
        wp_y = y_data[i]
        # 将输入的路径点按照分辨率进行插值，随后平滑处理
        reference_path = ReferencePath(map_data, wp_x, wp_y, resolution=0.1,
                                    smoothing_distance=5, max_width=30.0,
                                    circular=False)

        # 根据路径、地图数据，构建汽车模型
        car = BicycleModel(length=3, width=2,
                        reference_path=reference_path, Ts=0.1)

        # 创建MPC求解器
        N = 30
        Q = sparse.diags([1.0, 0.0, 0.0])
        R = sparse.diags([0.5, 0.0])
        QN = sparse.diags([1.0, 0.0, 0.0])

        v_max = 18.0  # m/s
        delta_max = 0.66  # rad
        ay_max = 15.0  # m/s^2
        InputConstraints = {'umin': np.array([0.0, -np.tan(delta_max)/car.length]),
                            'umax': np.array([v_max, np.tan(delta_max)/car.length])}
        StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                            'xmax': np.array([np.inf, np.inf, np.inf])}
        mpc = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)

        # 根据约束信息，计算路径速度
        a_min = -0.1  # m/s^2
        a_max = 0.5  # m/s^2
        SpeedProfileConstraints = {'a_min': a_min, 'a_max': a_max,
                                'v_min': 0.0, 'v_max': v_max, 'ay_max': ay_max}
        car.reference_path.compute_speed_profile(SpeedProfileConstraints)

        # 仿真运行记时
        t = 0.0
        index = 0

        # 仿真运行轨迹记录
        x_cache = []
        y_cache = []
        xy_cache = []

        while (car.s < reference_path.length):
            goal_x = reference_path.waypoints[-1].x
            goal_y = reference_path.waypoints[-1].y
            goal_distance = math.sqrt((goal_x-car.temporal_state.x)**2+(goal_y-car.temporal_state.y)**2)
            if(goal_distance<=5):
                print('goal tolerance achieved!')
                break

            plt.clf()
            # 记录汽车当前坐标
            x_cache.append(car.temporal_state.x)
            y_cache.append(car.temporal_state.y)

            # 计算障碍物距离以及方向单位向量
            distances = []
            orientations = []
            # 在地图上添加障碍物
            obs_to_map = []
            for obs_list in obs_cache:
                obs_x_list = obs_list[0]
                obs_y_list = obs_list[1]
                if(index<len(obs_x_list)):
                    obs_x = obs_x_list[index]
                    obs_y = obs_y_list[index]
                else:
                    obs_x = obs_x_list[-1]
                    obs_y = obs_y_list[-1]
                plt.scatter(obs_x, obs_y, s=100)
                #obs_to_map.append(Obstacle(cx=obs_x, cy=obs_y, radius=1))
                distance = math.sqrt((car.temporal_state.x -obs_x)**2+(car.temporal_state.y -obs_y)**2)
                distances.append(distance)
                orientations.append([(obs_x - car.temporal_state.x)/distance, (obs_y - car.temporal_state.y)/distance])

            # 更新地图
            #map_data.add_obstacles(obs_to_map)
            #car.reference_path.map = map_data

            # 人工势场法被动避障
            force = [0, 0]
            for d in range(len(distances)):
                distance = distances[d]
                v = calc_v(distance)
                force[0] += v*orientations[d][0]
                force[1] += v*orientations[d][1]

            # 如果周围没有其他车辆
            if(force[0] == 0 and force[1] == 0):
                # 通过mpc求解控制量
                u = mpc.get_control()
            # 如果周围有其他车辆
            else:
                # 更新mpc类内部状态
                mpc.model.get_current_waypoint()
                mpc.model.spatial_state = mpc.model.t2s(reference_state=
                    mpc.model.temporal_state, reference_waypoint=
                    mpc.model.current_waypoint)
                # 通过人工势场法进行被动避障
                vx = -force[0]
                vy = -force[1]
                v = math.sqrt(vx**2 + vy**2)
                psi = math.acos(vx/v)
                if(psi<=0):
                    psi = psi+2*math.pi
                delta_psi = psi - car.temporal_state.psi
                u = np.array([v*math.cos(delta_psi), delta_psi/car.Ts])

            # 汽车运动
            car.drive(u)
            # 记时
            t += car.Ts
            index += 1

            # 可视化路径
            reference_path.show()
            # 可视化汽车
            car.show()
            # 可视化mpc预测结果
            mpc.show_prediction()

            # 清除障碍物
            map_data.clear_obstacles(obs_to_map)

            plt.title('MPC Simulation: v(t): {:.2f}, delta(t): {:.2f}, Duration: '
                  '{:.2f} s'.format(u[0], u[1], t))
            plt.pause(0.00001)

        # 整合记录轨迹
        xy_cache = [x_cache, y_cache]
        obs_cache.append(xy_cache)