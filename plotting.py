"""
Plotting tools for Sampling-based algorithms
@author: huiming zhou
"""

from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

import numpy as np
import env
from linear_dynamic_model.LQR_CBF_rrtStar_linear import LQRrrtStar


class Plotting:
    def __init__(self, x_start, x_goal):
        self.xI, self.xG = x_start, x_goal
        self.env = env.Env()
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle

    def animation(self, nodelist, path, name, animation=False):
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)

    def animation_online(self, nodelist, name, animation=False):
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        plt.pause(1.0)
        plt.close()

    def animation_connect(self, V1, V2, path, name):
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)

    def plot_grid(self, name):
        fig, ax = plt.subplots()

        for ox, oy, w, h in self.obs_bound:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h, edgecolor="black", facecolor="black", fill=True
                )
            )

        for ox, oy, w, h in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h, edgecolor="black", facecolor="gray", fill=True
                )
            )

        for ox, oy, r in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r, edgecolor="black", facecolor="gray", fill=True
                )
            )

        plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        plt.plot(self.xG[0], self.xG[1], "rs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    @staticmethod
    def plot_visited(nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect(
                        "key_release_event",
                        lambda event: [exit(0) if event.key == "escape" else None],
                    )
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    @staticmethod
    def plot_visited_connect(V1, V2):
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].parent:
                    plt.plot([V1[k].x, V1[k].parent.x], [V1[k].y, V1[k].parent.y], "-g")
            if k < len2:
                if V2[k].parent:
                    plt.plot([V2[k].x, V2[k].parent.x], [V2[k].y, V2[k].parent.y], "-g")

            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot([x[0] for x in path], [x[1] for x in path], "-r", linewidth=2)
            plt.pause(0.01)
            plt.savefig("LQR-CBF_result.PNG")
        plt.show()

    @staticmethod
    def animation_dynamic(rrt_star: LQRrrtStar, name, dynamic_obs_initial):
        """Animation showing robot following path with moving obstacles"""
        fig, ax = plt.subplots(figsize=(12, 8))
        nodelist = rrt_star.vertex
        path = rrt_star.path
        
        # Plot boundaries
        for ox, oy, w, h in rrt_star.obs_boundary:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor="black", facecolor="black"))

        # Plot static obstacles
        for ox, oy, r in rrt_star.obs_circle:
            ax.add_patch(patches.Circle((ox, oy), r, edgecolor="black", facecolor="gray", alpha=0.7))

        # Plot tree
        for node in nodelist:
            if node.parent:
                ax.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g", alpha=0.2, linewidth=0.5)

        # Plot solution path
        if path:
            path_x = [x[0] for x in path]
            path_y = [x[1] for x in path]
            ax.plot(path_x, path_y, "-r", linewidth=2, label="Solution Path")

        # Plot start and goal
        ax.plot(rrt_star.s_start.x, rrt_star.s_start.y, "gs", markersize=10, label="Start")
        ax.plot(rrt_star.s_goal.x, rrt_star.s_goal.y, "rs", markersize=10, label="Goal")

        # Create robot
        robot_circle = None
        if path:
            path_reversed = path[::-1]
            robot_circle = patches.Circle((path_reversed[0][0], path_reversed[0][1]), 0.5, color='blue', alpha=0.8, zorder=5)
            ax.add_patch(robot_circle)

        # Create dynamic obstacles
        dynamic_circles = []
        if dynamic_obs_initial:
            for x, y, r, vx, vy in dynamic_obs_initial:
                circle = patches.Circle((x, y), r, color='orange', alpha=0.8, linewidth=2, edgecolor='black')
                ax.add_patch(circle)
                dynamic_circles.append(circle)

        ax.set_xlim(rrt_star.x_range)
        ax.set_ylim(rrt_star.y_range)
        ax.set_title(name)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        collision_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='red')

        print(f"Path length: {len(path)}")
        if path:
            print(f"Path element structure: {path[0]}")
            print(f"Path starts at: {path[-1]}, ends at: {path[0]}")

        def update(frame):
            current_time = frame * 0.05
            time_text.set_text(f'Time = {current_time:.2f}s')
            
            # Move robot along path
            if path and robot_circle:
                path_reversed = path[::-1]
                robot_x, robot_y = path_reversed[0][0], path_reversed[0][1]

                for i in range(len(path_reversed) - 1):
                    # If path has time info (3 elements), use it
                    if len(path_reversed[i]) >= 3:
                        t1 = path_reversed[i][2]
                        t2 = path_reversed[i+1][2]
                        
                        if t1 <= current_time <= t2:
                            # Interpolate between nodes based on time
                            alpha = (current_time - t1) / (t2 - t1) if t2 != t1 else 0
                            robot_x = path_reversed[i][0] * (1-alpha) + path_reversed[i+1][0] * alpha
                            robot_y = path_reversed[i][1] * (1-alpha) + path_reversed[i+1][1] * alpha
                            break
                        elif current_time > t2:
                            robot_x = path_reversed[i+1][0]
                            robot_y = path_reversed[i+1][1]
                
                robot_circle.center = (robot_x, robot_y)
            
            # Use stored obstacle positions if available, otherwise simulate
            for i, (x0, y0, r, vx, vy) in enumerate(dynamic_obs_initial):
                new_x = x0 + vx * current_time
                new_y = y0 + vy * current_time
                if i < len(dynamic_circles):
                    dynamic_circles[i].center = (new_x, new_y)
            
            # Check collisions visually
            collision_detected = False
            if 'robot_x' in locals() and dynamic_circles:
                for circle in dynamic_circles:
                    cx, cy = circle.center
                    dist = np.sqrt((robot_x - cx)**2 + (robot_y - cy)**2)
                    if dist < circle.radius + 0.5:
                        circle.set_edgecolor('red')
                        circle.set_linewidth(3)
                        collision_detected = True
                    else:
                        circle.set_edgecolor('black')
                        circle.set_linewidth(2)
            
            if collision_detected:
                collision_text.set_text('COLLISION DETECTED!')
            else:
                collision_text.set_text('')
            
            return [robot_circle, time_text, collision_text] + dynamic_circles

        anim = animation.FuncAnimation(fig, update, frames=600, interval=50, blit=True, repeat=True)
        plt.show()
        return anim