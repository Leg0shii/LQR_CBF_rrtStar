class Env:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

        self.dynamic_obs_circle = self.get_dynamic_obs_circle()
        self.time = 0.0

    def update_dynamic_obstacles(self, dt):
        """Update positions of dynamic obstacles"""
        self.time += dt
        for i in range(len(self.dynamic_obs_circle)):
            x, y, r, vx, vy = self.dynamic_obs_circle[i]
            # Simple linear motion (can be extended to more complex patterns)
            new_x = x + vx * dt
            new_y = y + vy * dt

            # Bounce off boundaries (optional)
            if new_x - r <= self.x_range[0] or new_x + r >= self.x_range[1]:
                vx = -vx
                new_x = x + vx * dt
            if new_y - r <= self.y_range[0] or new_y + r >= self.y_range[1]:
                vy = -vy
                new_y = y + vy * dt

            new_x = max(self.x_range[0] + r, min(new_x, self.x_range[1] - r))
            new_y = max(self.y_range[0] + r, min(new_y, self.y_range[1] - r))
            
            self.dynamic_obs_circle[i] = (new_x, new_y, r, vx, vy)

    def get_all_obstacles(self):
        """Get combined static and dynamic obstacles for collision checking"""
        all_circles = self.obs_circle.copy()
        # Add current positions of dynamic obstacles
        all_circles.extend([(x, y, r) for x, y, r, _, _ in self.dynamic_obs_circle])
        return all_circles, self.obs_rectangle, self.obs_boundary
    
    @staticmethod
    def get_dynamic_obs_circle():
        dynamic_obs_circle = [
            # Start with just 5-6 obstacles to test
            (8, 5, 2, 0.2, 0.1),      # Near start
            (20, 10, 2, -0.1, 0.2),    # Middle
            (30, 15, 2, 0.1, -0.1),    # Center-right
            (40, 20, 2, -0.2, 0.1),    # Near goal
            (25, 8, 2, 0.15, 0.15),    # Lower middle
        ]
        return dynamic_obs_circle

    @staticmethod
    def obs_boundary():  # circle
        obs_boundary = [[0, 0, 1, 30], [0, 30, 50, 1], [1, 0, 50, 1], [50, 1, 1, 30]]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        # obs_rectangle = [
        #     [14, 12, 8, 2],
        #     [18, 22, 8, 3],
        #     [26, 7, 2, 12],
        #     [32, 14, 10, 2]
        # ]
        obs_rectangle = []
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [7, 12, 3],
            [46, 10, 2],
            [25, 10, 3],
            [15, 5, 2],
            [15, 15, 2],
            [37, 7, 3],
            [37, 23, 3],
        ]

        return obs_cir
