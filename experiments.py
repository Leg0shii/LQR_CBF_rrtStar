import contextlib
import csv
import io
import itertools
import math
import sys
import time
import numpy as np
from scipy import stats
import random
import copy
import logging
from datetime import datetime
import json
import os

from linear_dynamic_model.LQR_CBF_rrtStar_linear import LQRrrtStar

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'collision_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

os.environ['GUROBI_LOG_LEVEL'] = '0'
logging.getLogger('gurobipy').setLevel(logging.ERROR)


# Suppress all Gurobi output
@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# Import Gurobi silently
with suppress_stdout():
    import gurobipy as gp
    # Set Gurobi to quiet mode globally
    gp.setParam('OutputFlag', 0)

def save_collision_trajectory(rollout_num, path, dynamic_obstacles, collision_info, x_start, x_goal, timestamp=None):
    """
    Save collision trajectory and details to a JSON file for replay
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    collision_dir = "collision_trajectories"
    if not os.path.exists(collision_dir):
        os.makedirs(collision_dir)
    
    trajectory_data = {
        "rollout_number": rollout_num,
        "timestamp": timestamp,
        "start_position": list(x_start),
        "goal_position": list(x_goal),
        "collision_info": {
            "collision_time": collision_info['time'],
            "collision_obstacle_idx": collision_info['obstacle_idx'],
            "collision_distance": collision_info['distance'],
            "collision_path_point_idx": collision_info['path_point'],
            "robot_position_at_collision": list(collision_info['robot_pos']),
            "obstacle_position_at_collision": list(collision_info['obstacle_pos'])
        },
        "dynamic_obstacles": [
            {
                "initial_x": obs[0],
                "initial_y": obs[1],
                "radius": obs[2],
                "velocity_x": obs[3],
                "velocity_y": obs[4],
                "speed": math.sqrt(obs[3]**2 + obs[4]**2)
            } for obs in dynamic_obstacles
        ],
        "robot_trajectory": [
            {
                "x": point[0],
                "y": point[1],
                "time": point[2] if len(point) >= 3 else 0,
                "waypoint_idx": i
            } for i, point in enumerate(path[::-1])  # Reverse to get start-to-goal order
        ],
        "collision_details": {
            "robot_x_at_collision": collision_info['robot_pos'][0],
            "robot_y_at_collision": collision_info['robot_pos'][1],
            "obstacle_x_at_collision": collision_info['obstacle_pos'][0],
            "obstacle_y_at_collision": collision_info['obstacle_pos'][1],
            "obstacle_initial_position": {
                "x": dynamic_obstacles[collision_info['obstacle_idx']][0],
                "y": dynamic_obstacles[collision_info['obstacle_idx']][1],
                "radius": dynamic_obstacles[collision_info['obstacle_idx']][2]
            },
            "obstacle_velocity": {
                "vx": dynamic_obstacles[collision_info['obstacle_idx']][3],
                "vy": dynamic_obstacles[collision_info['obstacle_idx']][4]
            }
        },
        "environment_bounds": {
            "x_min": 5,
            "x_max": 45,
            "y_min": 5,
            "y_max": 25
        }
    }
    
    filename = f"{collision_dir}/collision_rollout_{rollout_num}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    logging.info(f"Saved collision trajectory to {filename}")
    logging.info(f"Collision Details:")
    logging.info(f"  - Time of collision: {collision_info['time']:.3f} seconds")
    logging.info(f"  - Robot position: ({collision_info['robot_pos'][0]:.2f}, {collision_info['robot_pos'][1]:.2f})")
    logging.info(f"  - Obstacle {collision_info['obstacle_idx']} position: ({collision_info['obstacle_pos'][0]:.2f}, {collision_info['obstacle_pos'][1]:.2f})")
    logging.info(f"  - Distance at collision: {collision_info['distance']:.3f}")
    
    return filename

def save_collision_summary(all_collisions, timestamp=None):
    """
    Save a summary of all collisions for quick analysis
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    collision_dir = "collision_trajectories"
    if not os.path.exists(collision_dir):
        os.makedirs(collision_dir)
    
    summary_file = f"{collision_dir}/collision_summary_{timestamp}.json"
    
    summary_data = {
        "timestamp": timestamp,
        "total_collisions": len(all_collisions),
        "collisions": all_collisions
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logging.info(f"Saved collision summary to {summary_file}")
    return summary_file

def is_goal_accessible(goal, dynamic_obstacles, max_time=100.0):
    """Check if goal will be covered by obstacles during execution"""
    gx, gy = goal
    for t in np.arange(0, max_time, 1.0):
        for obs in dynamic_obstacles:
            x0, y0, r, vx, vy = obs
            # Predict obstacle position at time t
            obs_x = x0 + vx * t
            obs_y = y0 + vy * t
            
            # Check if obstacle covers goal
            dist = math.hypot(gx - obs_x, gy - obs_y)
            if dist < r + 3.0:
                return False
    return True

def is_point_feasible(point, dynamic_obstacles, static_margin=2.0):
    """Check if a point is not inside any obstacle"""
    x, y = point
    
    for obs in dynamic_obstacles:
        ox, oy, r = obs[0], obs[1], obs[2]
        if math.hypot(x - ox, y - oy) < r + static_margin:
            return False
    
    if x < 5 or x > 45 or y < 5 or y > 25:
        return False
    
    return True

def run_collision_experiment(num_rollouts=3000, max_speed=1.0, cbf_enabled=True, save_rq2_data=False, save_rq3_data=False, alpha=None, v_nom=5.0, step_len=10):
    """
    Run multiple rollouts to determine collision rate with 95% confidence interval
    """
    collision_count = 0
    collision_details = []
    collision_files = []
    successful_paths = 0
    skipped_rollouts = 0
    
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.info(f"Starting experiment with {num_rollouts} rollouts, max_speed={max_speed} m/s")
    logging.info(f"Experiment timestamp: {experiment_timestamp}")
    
    rq2_data = [] if save_rq2_data else None
    rq3_data = [] if save_rq3_data else None
    for rollout in range(num_rollouts):
        rollout_header = f"\n--- ROLLOUT {rollout + 1}/{num_rollouts} ---"
        print(rollout_header)
        logging.info(rollout_header)
        
        if rollout % 100 == 0 and rollout > 0:
            current_rate = collision_count / rollout
            progress_msg = f"Progress: {rollout}/{num_rollouts} rollouts completed, current collision rate: {current_rate:.4f} ({current_rate*100:.2f}%)"
            logging.info(progress_msg)
        
        # Keep trying until we get a valid configuration
        valid_config = False
        config_attempts = 0
        max_config_attempts = 50
        
        while not valid_config and config_attempts < max_config_attempts:
            config_attempts += 1
            
            num_obstacles = random.randint(3, 10)
            dynamic_obstacles = []
            
            for i in range(num_obstacles):
                placed = False
                for _ in range(20):  # Try 20 times to place each obstacle
                    x = random.uniform(15, 35)
                    y = random.uniform(12, 18)
                    r = random.uniform(1.5, 2.5)
                    
                    # Check minimum distance from other obstacles
                    too_close = False
                    for ox, oy, or_, _, _ in dynamic_obstacles:
                        if math.hypot(x - ox, y - oy) < r + or_ + 2.0:
                            too_close = True
                            break
                    
                    if not too_close:
                        angle = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(0.2, max_speed)
                        vx = speed * math.cos(angle)
                        vy = speed * math.sin(angle)
                        dynamic_obstacles.append((x, y, r, vx, vy))
                        placed = True
                        break
                
                if not placed and i > 2:
                    break
            
            start_found = False
            for _ in range(50):
                x_start = (random.uniform(5, 45), random.uniform(5, 25))
                if is_point_feasible(x_start, dynamic_obstacles):
                    start_found = True
                    break
            
            if not start_found:
                continue
            
            # Find goal position that stays accessible
            goal_found = False
            for _ in range(50):
                x_goal = (random.uniform(5, 45), random.uniform(5, 25))
                dist_to_start = math.hypot(x_goal[0] - x_start[0], x_goal[1] - x_start[1])
                
                if (is_point_feasible(x_goal, dynamic_obstacles) and 
                    dist_to_start >= 15 and
                    is_goal_accessible(x_goal, dynamic_obstacles)):
                    goal_found = True
                    break
            
            if start_found and goal_found:
                valid_config = True
        
        if not valid_config:
            skip_msg = f"  SKIPPED: Could not find valid configuration after {config_attempts} attempts"
            print(skip_msg)
            logging.warning(skip_msg)
            skipped_rollouts += 1
            continue
        
        obstacles_msg = f"Obstacles ({len(dynamic_obstacles)}):"
        print(obstacles_msg)
        logging.info(obstacles_msg)
        
        for i, obs in enumerate(dynamic_obstacles):
            x, y, r, vx, vy = obs
            speed = math.sqrt(vx**2 + vy**2)
            obs_msg = f"  Obs {i}: pos=({x:.1f}, {y:.1f}), r={r:.1f}, v=({vx:.2f}, {vy:.2f}), speed={speed:.2f}"
            print(obs_msg)
            logging.info(obs_msg)
        
        start_msg = f"Start: ({x_start[0]:.1f}, {x_start[1]:.1f})"
        goal_msg = f"Goal:  ({x_goal[0]:.1f}, {x_goal[1]:.1f})"
        dist_msg = f"Distance: {math.hypot(x_goal[0] - x_start[0], x_goal[1] - x_start[1]):.1f}"
        
        print(start_msg)
        print(goal_msg)
        print(dist_msg)
        logging.info(start_msg)
        logging.info(goal_msg)
        logging.info(dist_msg)
        
        rrt_star = LQRrrtStar(
            x_start, x_goal, 
            step_len=step_len, 
            goal_sample_rate=0.10, 
            search_radius=20, 
            iter_max=1500,
            solve_QP=cbf_enabled and save_rq2_data
        )

        if save_rq3_data and alpha is not None:
            rrt_star.lqr_planner.cbf_rrt_simulation.k_cbf = alpha
        if save_rq3_data and v_nom is not None:
            rrt_star.nominal_velocity = v_nom

        if not cbf_enabled and save_rq2_data:
            original_lqr_planning = rrt_star.lqr_planner.lqr_planning
            def modified_lqr_planning(sx, sy, gx, gy, test_LQR=False, show_animation=True, cbf_check=True, solve_QP=False, current_time=0.0, time_horizon=0.5):
                return original_lqr_planning(sx, sy, gx, gy, test_LQR=test_LQR, show_animation=show_animation, cbf_check=False, solve_QP=False, current_time=current_time, time_horizon=time_horizon)
            
            rrt_star.lqr_planner.lqr_planning = modified_lqr_planning
        
        # Override the dynamic obstacles
        rrt_star.env.dynamic_obs_circle = dynamic_obstacles
        rrt_star.initial_dynamic_obs = copy.deepcopy(dynamic_obstacles)
        rrt_star.lqr_planner.cbf_rrt_simulation.dynamic_obstacles = dynamic_obstacles
        rrt_star.utils.dynamic_obs_circle = dynamic_obstacles
        
        index = None
        nodes_added = 0
        t0 = time.perf_counter()
        for k in range(rrt_star.iter_max):
            rrt_star.planning_time += rrt_star.dt
            node_rand = rrt_star.generate_random_node(rrt_star.goal_sample_rate)
            node_rand.time = rrt_star.planning_time
            node_near = rrt_star.nearest_neighbor(rrt_star.vertex, node_rand)
            
            dist_to_new = math.hypot(node_rand.x - node_near.x, node_rand.y - node_near.y)
            time_to_reach = min(dist_to_new, rrt_star.step_len) / v_nom
            node_rand.time = node_near.time + time_to_reach
            
            node_new = rrt_star.LQR_steer(node_near, node_rand)
            
            if node_new and not rrt_star.utils.is_collision_with_dynamic_predicted(
                node_near, node_new, rrt_star.initial_dynamic_obs
            ):
                neighbor_index = rrt_star.find_near_neighbor(node_new)
                rrt_star.vertex.append(node_new)
                nodes_added += 1
                if neighbor_index:
                    rrt_star.LQR_choose_parent(node_new, neighbor_index)
                    rrt_star.rewire(node_new, neighbor_index)
        
        planning_msg = f"Planning: {nodes_added} nodes added, tree size: {len(rrt_star.vertex)}"
        planning_time_ms = (time.perf_counter() - t0) * 1000
        print(planning_msg)
        logging.info(planning_msg)
        
        index = rrt_star.search_goal_parent()
        
        if index is None:
            collision_count += 1
            no_path_info = {
                'time': 0,
                'obstacle_idx': -1,
                'distance': 0,
                'path_point': 0,
                'robot_pos': x_start,
                'obstacle_pos': (0, 0)
            }
            
            # Create empty path for no-path-found scenarios
            empty_path = [(x_start[0], x_start[1], 0)]
            
            collision_file = save_collision_trajectory(
                rollout, empty_path, dynamic_obstacles, no_path_info,
                x_start, x_goal, experiment_timestamp
            )
            
            collision_details.append({
                'rollout': rollout,
                'reason': 'no_path_found',
                'num_obstacles': len(dynamic_obstacles),
                'nodes_in_tree': len(rrt_star.vertex),
                'trajectory_file': collision_file
            })
            collision_files.append(collision_file)
            
            result_msg = f"Result: NO PATH FOUND"
            print(result_msg)
            logging.warning(result_msg)
            continue
        
        path, _ = rrt_star.extract_path(rrt_star.vertex[index])
        path_msg = f"Path: {len(path)} waypoints"
        print(path_msg)
        logging.info(path_msg)
        
        # Simulate robot following the path and check for collisions
        collision_detected, collision_info = check_path_for_collisions_detailed(path, dynamic_obstacles)
        if collision_detected:
            collision_count += 1
            
            # Save the collision trajectory
            collision_file = save_collision_trajectory(
                rollout, path, dynamic_obstacles, collision_info,
                x_start, x_goal, experiment_timestamp
            )
            
            collision_details.append({
                'rollout': rollout,
                'reason': 'collision_on_path',
                'num_obstacles': len(dynamic_obstacles),
                'collision_info': collision_info,
                'trajectory_file': collision_file
            })
            collision_files.append(collision_file)
            
            result_msg = f"Result: COLLISION at t={collision_info['time']:.2f}s with obstacle {collision_info['obstacle_idx']}"
            print(result_msg)
            print(f"  Saved trajectory to: {collision_file}")
            logging.warning(result_msg)
        else:
            successful_paths += 1
            result_msg = f"Result: SUCCESS"
            print(result_msg)
            logging.info(result_msg)

        if save_rq2_data:
            rq2_data.append({
                'rollout': rollout,
                'cbf': int(cbf_enabled),
                'T_ms': planning_time_ms,
                'J': rrt_star.path_cost(path) if path else float('inf'),
                'success': int(index is not None),
                'nodes': len(rrt_star.vertex)
            })

        if save_rq3_data:
            min_signed_dist = calculate_min_signed_distance(path, dynamic_obstacles) if index is not None else float('inf')
            rq3_data.append({
                'rollout': rollout,
                'success': int(index is not None),
                'min_signed_dist': min_signed_dist
            })
        
        del rrt_star
        if rollout % 10 == 0:
            import gc
            gc.collect()
    
    if collision_files:
        summary_file = save_collision_summary(collision_details, experiment_timestamp)
        print(f"\nCollision summary saved to: {summary_file}")
    
    # Calculate statistics
    actual_rollouts = num_rollouts - skipped_rollouts
    collision_rate = collision_count / actual_rollouts if actual_rollouts > 0 else 0
    
    # Wilson score interval for 95% confidence
    confidence_level = 0.95
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    n = actual_rollouts
    p_hat = collision_rate
    
    # Wilson score interval formula
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4*n**2))) / denominator
    
    lower_bound = center - margin
    upper_bound = center + margin
    
    # Print and log results
    results = f"""
=== COLLISION RATE ANALYSIS ===
Total rollouts attempted: {num_rollouts}
Rollouts skipped (infeasible start/goal): {skipped_rollouts}
Actual rollouts completed: {actual_rollouts}
Max obstacle speed: {max_speed} m/s

Results:
- Successful paths: {successful_paths}
- Collisions detected: {collision_count}
  * No path found: {sum(1 for d in collision_details if d['reason'] == 'no_path_found')}
  * Collision on path: {sum(1 for d in collision_details if d['reason'] == 'collision_on_path')}
- Collision rate: {collision_rate:.4f} ({collision_rate*100:.2f}%)
- 95% Wilson confidence interval: [{lower_bound:.4f}, {upper_bound:.4f}]
- 95% Wilson upper bound: {upper_bound:.4f} ({upper_bound*100:.2f}%)

Collision trajectories saved: {len(collision_files)} files
Directory: collision_trajectories/

Verdict: {'PASSED' if upper_bound < 0.01 else 'FAILED'}: 95% upper bound {'is below' if upper_bound < 0.01 else 'exceeds'} 1%
"""
    
    print(results)
    logging.info(results)

    if save_rq2_data:
        with open(f'rq2_data_cbf{int(cbf_enabled)}.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['rollout', 'cbf', 'T_ms', 'J', 'success', 'nodes'])
            writer.writeheader()
            for row in rq2_data:
                writer.writerow(row)
    
    if save_rq3_data:
        return collision_rate, lower_bound, upper_bound, collision_details, rq3_data
    else:
        return collision_rate, lower_bound, upper_bound, collision_details

def check_path_for_collisions_detailed(path, dynamic_obstacles):
    """
    Check if a POINT robot actually collides with any dynamic obstacle.
    """
    if not path:
        return True, {'time': 0, 'obstacle_idx': -1, 'distance': 0}
    
    path_forward = path[::-1]
    
    interpolated_path = []
    for i in range(len(path_forward) - 1):
        p1 = path_forward[i]
        p2 = path_forward[i + 1]
        
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_interp = max(2, int(dist / 0.05))  # Check every 0.05m
        
        for j in range(num_interp):
            alpha = j / num_interp
            x = p1[0] + alpha * (p2[0] - p1[0])
            y = p1[1] + alpha * (p2[1] - p1[1])
            t = p1[2] + alpha * (p2[2] - p1[2]) if len(p1) > 2 and len(p2) > 2 else 0
            interpolated_path.append((x, y, t, i))
    
    if path_forward:
        last = path_forward[-1]
        interpolated_path.append((last[0], last[1], last[2] if len(last) > 2 else 0, len(path_forward)-1))
    
    # Check for collisions along interpolated path
    for point in interpolated_path:
        robot_x, robot_y, t, original_idx = point[0], point[1], point[2], point[3]
        
        for obs_idx, obs in enumerate(dynamic_obstacles):
            if len(obs) >= 5:
                x0, y0, obs_radius, vx, vy = obs[:5]
                
                obs_x = x0 + vx * t
                obs_y = y0 + vy * t
                
                distance_to_center = math.hypot(robot_x - obs_x, robot_y - obs_y)
                
                # COLLISION: Point robot is inside obstacle circle
                if distance_to_center < obs_radius:
                    return True, {
                        'time': t,
                        'obstacle_idx': obs_idx,
                        'distance': distance_to_center,
                        'obstacle_radius': obs_radius,
                        'penetration_depth': obs_radius - distance_to_center,
                        'path_point': original_idx,
                        'robot_pos': (robot_x, robot_y),
                        'obstacle_pos': (obs_x, obs_y)
                    }
    
    return False, None

def calculate_min_signed_distance(path, dynamic_obstacles):
    """
    Calculate minimum signed distance along path.
    Positive = safe distance, Negative = collision/penetration
    """
    if not path:
        return float('inf')
    
    path_forward = path[::-1]
    min_signed_distance = float('inf')
    
    for i in range(len(path_forward) - 1):
        p1 = path_forward[i]
        p2 = path_forward[i + 1]
        
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_interp = max(2, int(dist / 0.1))
        
        for j in range(num_interp + 1):
            alpha = j / num_interp
            x = p1[0] + alpha * (p2[0] - p1[0])
            y = p1[1] + alpha * (p2[1] - p1[1])
            t = p1[2] + alpha * (p2[2] - p1[2]) if len(p1) > 2 and len(p2) > 2 else 0
            
            for obs in dynamic_obstacles:
                if len(obs) >= 5:
                    x0, y0, obs_radius, vx, vy = obs[:5]
                    obs_x = x0 + vx * t
                    obs_y = y0 + vy * t
                    center_dist = math.hypot(x - obs_x, y - obs_y)
                    signed_dist = center_dist - obs_radius
                    min_signed_distance = min(min_signed_distance, signed_dist)
    
    return min_signed_distance


def run_rq3_experiments(num_rollouts=100):
    """Run all RQ3 parameter combinations"""
    alpha_values = [0.25, 0.5, 1.0]
    v_nom_values = [1.5, 3.0, 5.0]
    step_len_values = [6, 12]
    
    all_results = []
    
    for alpha, v_nom, step_len in itertools.product(alpha_values, v_nom_values, step_len_values):
        print(f"\n=== Testing: Alpha={alpha}, V_nom={v_nom}, Step_len={step_len} ===")
        
        collision_rate, lower, upper, details, rq3_data = run_collision_experiment(
            num_rollouts=num_rollouts,
            max_speed=2.0,
            cbf_enabled=True,
            save_rq3_data=True,
            alpha=alpha,
            v_nom=v_nom,
            step_len=step_len
        )
        
        # Compute statistics from the returned data
        success_rate = sum(row['success'] for row in rq3_data) / len(rq3_data) if rq3_data else 0
        successful_runs = [row for row in rq3_data if row['success']]
        min_dists = [row['min_signed_dist'] for row in successful_runs if row['min_signed_dist'] != float('inf')]
        avg_min_dist = np.mean(min_dists) if min_dists else None
        std_min_dist = np.std(min_dists) if min_dists else None
        
        result = {
            'alpha': alpha,
            'v_nom': v_nom, 
            'step_len': step_len,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'avg_min_dist': avg_min_dist,
            'std_min_dist': std_min_dist,
            'num_success': len(successful_runs),
            'num_total': len(rq3_data)
        }
        all_results.append(result)
        
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Collision rate: {collision_rate:.2%}")
        if avg_min_dist:
            print(f"  Avg min distance: {avg_min_dist:.3f} ± {std_min_dist:.3f}")
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f'rq3_summary_{timestamp}.csv'
    with open(summary_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['alpha', 'v_nom', 'step_len', 'success_rate', 
                                               'collision_rate', 'avg_min_dist', 'std_min_dist',
                                               'num_success', 'num_total'])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n=== RQ3 SUMMARY (saved to {summary_file}) ===")
    print(f"{'Alpha':<8} {'V_nom':<8} {'Step':<8} {'Success':<12} {'Collision':<12} {'Min Dist':<20}")
    print("-" * 70)
    for r in all_results:
        min_dist_str = f"{r['avg_min_dist']:.3f} ± {r['std_min_dist']:.3f}" if r['avg_min_dist'] else "N/A"
        print(f"{r['alpha']:<8.1f} {r['v_nom']:<8.1f} {r['step_len']:<8} "
              f"{r['success_rate']:<12.2%} {r['collision_rate']:<12.2%} {min_dist_str}")


if __name__ == "__main__":
    rq_num = sys.argv[1]
    if rq_num == "1":
        collision_rate, lower, upper, details = run_collision_experiment(
            num_rollouts=3000,
            max_speed=1.0,
            cbf_enabled=True
        )

    elif rq_num == "2":
        print("Running RQ2 comparison...")
        
        random.seed(42)
        np.random.seed(42)
        run_collision_experiment(num_rollouts=100, cbf_enabled=True, save_rq2_data=True)
        
        random.seed(42)
        np.random.seed(42)
        run_collision_experiment(num_rollouts=100, cbf_enabled=False, save_rq2_data=True)
        
        print("RQ2 data saved to rq2_data_cbf1.csv and rq2_data_cbf0.csv")

    elif rq_num == "3":
        print("Running RQ3 parameter sensitivity analysis...")
        run_rq3_experiments(num_rollouts=100)
    else:
        print(f"Unknown RQ number: {rq_num}")
        print("Usage: python script.py [1|2|3]")