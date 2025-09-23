import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import sys
import os
import glob

def load_collision_trajectory(filename):
    """Load collision trajectory from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def replay_collision(trajectory_data, save_animation=False, output_file=None, debug=False):
    """
    Replay a collision trajectory with animation
    """
    robot_traj = trajectory_data['robot_trajectory']
    obstacles = trajectory_data['dynamic_obstacles']
    collision_info = trajectory_data['collision_info']
    bounds = trajectory_data['environment_bounds']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(bounds['x_min'], bounds['x_max'])
    ax.set_ylim(bounds['y_min'], bounds['y_max'])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    title = f"Rollout {trajectory_data['rollout_number']} - Collision at t={collision_info['collision_time']:.2f}s"
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    start = trajectory_data['start_position']
    goal = trajectory_data['goal_position']
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    
    path_x = [p['x'] for p in robot_traj]
    path_y = [p['y'] for p in robot_traj]
    ax.plot(path_x, path_y, 'b--', alpha=0.3, label='Planned Path')
    
    max_time = max([p['time'] for p in robot_traj]) if robot_traj else collision_info['collision_time']
    for i, obs in enumerate(obstacles):
        traj_times = [0, max_time * 1.2]
        traj_x = [obs['initial_x'] + obs['velocity_x'] * t for t in traj_times]
        traj_y = [obs['initial_y'] + obs['velocity_y'] * t for t in traj_times]
        
        if i == collision_info['collision_obstacle_idx']:
            ax.plot(traj_x, traj_y, 'r-', alpha=0.2, linewidth=2)
        else:
            ax.plot(traj_x, traj_y, 'gray', alpha=0.1, linewidth=1)
    
    robot_circle = Circle((start[0], start[1]), 0.5, color='blue', fill=True, alpha=0.7)
    ax.add_patch(robot_circle)
    
    obstacle_circles = []
    obstacle_velocity_arrows = []
    for i, obs in enumerate(obstacles):
        circle = Circle((obs['initial_x'], obs['initial_y']), obs['radius'], color='red', fill=True, alpha=0.5)
        ax.add_patch(circle)
        obstacle_circles.append(circle)
        
        # Add velocity arrow for each obstacle
        arrow = ax.arrow(
            obs['initial_x'],
            obs['initial_y'],
            obs['velocity_x'] * 2,
            obs['velocity_y'] * 2,
            head_width=0.3,
            head_length=0.2,
            fc='red',
            ec='red',
            alpha=0.3
        )
        obstacle_velocity_arrows.append(arrow)
    
    collision_marker = ax.plot([], [], 'rx', markersize=20, markeredgewidth=3, label='Collision')[0]
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    ax.legend(loc='upper right')
    
    def init():
        robot_circle.center = (start[0], start[1])
        for i, circle in enumerate(obstacle_circles):
            circle.center = (obstacles[i]['initial_x'], obstacles[i]['initial_y'])
        collision_marker.set_data([], [])
        time_text.set_text('')
        return [robot_circle] + obstacle_circles + [collision_marker, time_text]
    
    def animate(frame):
        if frame < len(robot_traj):
            current_point = robot_traj[frame]
            current_time = current_point['time']
            robot_x = current_point['x']
            robot_y = current_point['y']
        else:
            current_time = collision_info['collision_time']
            robot_x = collision_info['robot_position_at_collision'][0]
            robot_y = collision_info['robot_position_at_collision'][1]
        
        robot_circle.center = (robot_x, robot_y)
        
        if debug and frame % 10 == 0:
            print(f"\nFrame {frame}, Time: {current_time:.3f}s")
            print(f"Robot: ({robot_x:.2f}, {robot_y:.2f})")
        
        for i, circle in enumerate(obstacle_circles):
            obs = obstacles[i]
            obs_x = obs['initial_x'] + obs['velocity_x'] * current_time
            obs_y = obs['initial_y'] + obs['velocity_y'] * current_time
            circle.center = (obs_x, obs_y)
            
            if debug and frame % 10 == 0 and i == collision_info['collision_obstacle_idx']:
                print(f"Obstacle {i}: ({obs_x:.2f}, {obs_y:.2f}) = "
                      f"({obs['initial_x']:.2f}, {obs['initial_y']:.2f}) + "
                      f"({obs['velocity_x']:.2f}, {obs['velocity_y']:.2f}) * {current_time:.2f}")
            
            if i == collision_info['collision_obstacle_idx'] and current_time >= collision_info['collision_time']:
                circle.set_color('darkred')
                circle.set_alpha(0.8)
            else:
                circle.set_color('red')
                circle.set_alpha(0.5)
        
        if current_time >= collision_info['collision_time']:
            collision_marker.set_data(
                [collision_info['robot_position_at_collision'][0]], 
                [collision_info['robot_position_at_collision'][1]]
            )
            robot_circle.set_color('darkblue')
            robot_circle.set_alpha(0.9)
        else:
            collision_marker.set_data([], [])
            robot_circle.set_color('blue')
            robot_circle.set_alpha(0.7)
        
        time_text.set_text(f'Time: {current_time:.2f}s')
        return [robot_circle] + obstacle_circles + [collision_marker, time_text]
    
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(robot_traj) + 30,
        interval=50, blit=True, repeat=True
    )
    
    if save_animation and output_file:
        anim.save(output_file, writer='pillow', fps=20)
        print(f"Animation saved to {output_file}")
    
    plt.show()
    
    return fig, anim

def print_collision_details(trajectory_data):
    """Print detailed collision information"""
    collision_info = trajectory_data['collision_info']
    collision_details = trajectory_data['collision_details']
    
    print("\n" + "="*50)
    print(f"COLLISION DETAILS - Rollout {trajectory_data['rollout_number']}")
    print("="*50)
    
    print(f"\nCollision Time: {collision_info['collision_time']:.3f} seconds")
    print(f"Collision occurred at waypoint {collision_info['collision_path_point_idx']} of {len(trajectory_data['robot_trajectory'])}")
    
    print(f"\nRobot Position at Collision:")
    print(f"  X: {collision_details['robot_x_at_collision']:.3f} m")
    print(f"  Y: {collision_details['robot_y_at_collision']:.3f} m")
    
    print(f"\nObstacle {collision_info['collision_obstacle_idx']} Position at Collision:")
    print(f"  X: {collision_details['obstacle_x_at_collision']:.3f} m")
    print(f"  Y: {collision_details['obstacle_y_at_collision']:.3f} m")
    
    print(f"\nObstacle Initial Configuration:")
    print(f"  Initial Position: ({collision_details['obstacle_initial_position']['x']:.3f}, "
          f"{collision_details['obstacle_initial_position']['y']:.3f}) m")
    print(f"  Radius: {collision_details['obstacle_initial_position']['radius']:.3f} m")
    print(f"  Velocity: ({collision_details['obstacle_velocity']['vx']:.3f}, "
          f"{collision_details['obstacle_velocity']['vy']:.3f}) m/s")
    
    print(f"\nDistance at collision: {collision_info['collision_distance']:.3f} m")
    
    print("\nEnvironment Configuration:")
    print(f"  Start: ({trajectory_data['start_position'][0]:.2f}, {trajectory_data['start_position'][1]:.2f})")
    print(f"  Goal: ({trajectory_data['goal_position'][0]:.2f}, {trajectory_data['goal_position'][1]:.2f})")
    print(f"  Number of obstacles: {len(trajectory_data['dynamic_obstacles'])}")

def list_collision_files():
    """List all available collision trajectory files"""
    collision_dir = "collision_trajectories"
    if not os.path.exists(collision_dir):
        print("No collision_trajectories directory found.")
        return []
    
    files = glob.glob(f"{collision_dir}/collision_rollout_*.json")
    files.sort()
    
    if not files:
        print("No collision files found.")
        return []
    
    print("\nAvailable collision trajectories:")
    print("-" * 50)
    for i, file in enumerate(files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    return files

def main():
    """Main function to replay collision trajectories"""
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        files = list_collision_files()
        if not files:
            return
        
        try:
            choice = input("\nEnter file number to replay (or 'all' for all files): ").strip()
            
            if choice.lower() == 'all':
                for file in files:
                    print(f"\nReplaying: {file}")
                    data = load_collision_trajectory(file)
                    print_collision_details(data)
                    replay_collision(data)
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    filename = files[idx]
                else:
                    print("Invalid choice.")
                    return
        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")
            return
    
    if 'filename' in locals():
        print(f"\nLoading: {filename}")
        trajectory_data = load_collision_trajectory(filename)
        
        print_collision_details(trajectory_data)
        
        save_anim = input("\nSave animation as GIF? (y/n): ").strip().lower() == 'y'
        output_file = None
        if save_anim:
            output_file = filename.replace('.json', '.gif')
        
        replay_collision(trajectory_data, save_animation=save_anim, output_file=output_file)

if __name__ == "__main__":
    main()