import torch
import numpy as np

class mapping:
    def __init__(self, **parameter):
        """
        Vanishing-Point-based Pipe mapping code
        **parameter
        f = camera focal length
        cx = camera center_x
        cy = camera center_y
        pipe_diameter = mm
        pixel_over_mm = pixel/mm
        mapping_max_L = maximum mapping L
        """
        self.water = parameter.get("water",False)
        self.f = parameter.get("f", 934.89)
        if self.water == True:
            self.f = self.f*1.33
        self.cx = parameter.get("cx", 640)
        self.cy = parameter.get("cy", 360)
        self.diameter = parameter.get("pipe_diameter", 84.6)
        self.mm_over_pixel = parameter.get("pixel_over_mm", 0.1)
        self.mapping_max_L = parameter.get("mapping_max_L", 300)
        self.weight = 1 / self.mm_over_pixel
        self.radius = self.diameter / 2
        self.device = parameter.get("device", "cuda")
        self.points_3D = self.create_3D_points(self.mapping_max_L)

        
    def create_3D_points(self, max_L):
        self.L_num = int(self.mapping_max_L * self.weight)
        self.R_num = int(self.diameter * np.pi * self.weight)
        z_values = np.linspace(0, self.mapping_max_L, self.L_num)
        angles = np.linspace(0, 2 * np.pi, self.R_num)

        x_coords = self.radius * np.cos(angles)
        y_coords = self.radius * np.sin(angles)

        x_grid, z_grid = np.meshgrid(x_coords, z_values)
        y_grid, _ = np.meshgrid(y_coords, z_values)

        points = np.stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), np.ones(x_grid.size)), axis=1)

        return torch.tensor(points, dtype=torch.float32, device=self.device)

    def project_points_to_2d(self, points, fx, fy, cx, cy, alpha, beta, gamma, tx, ty, tz):
        # Camera intrinsic matrix K
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # Rotation matrices for roll, pitch, yaw
        R_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(alpha), -torch.sin(alpha)],
            [0, torch.sin(alpha), torch.cos(alpha)]
        ], dtype=torch.float32, device=self.device)

        R_y = torch.tensor([
            [torch.cos(beta), 0, torch.sin(beta)],
            [0, 1, 0],
            [-torch.sin(beta), 0, torch.cos(beta)]
        ], dtype=torch.float32, device=self.device)

        # Ensure gamma is a Tensor
        gamma = torch.tensor(gamma, dtype=torch.float32, device=self.device)

        R_z = torch.tensor([
            [torch.cos(gamma), -torch.sin(gamma), 0],
            [torch.sin(gamma), torch.cos(gamma), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # Combined rotation matrix
        R = R_z @ R_y @ R_x
        T = torch.tensor([tx, ty, tz], dtype=torch.float32, device=self.device)

        # Apply rotation and translation
        points_cam = torch.matmul(points[:, :3], R.T) + T

        # Project to 2D
        points_proj = torch.matmul(points_cam, K.T)
        points_2d = points_proj[:, :2] / points_proj[:, 2:3]

        return points_2d.to(torch.int)


    def find_tx_ty(self, step, radius, degrees):
        # Convert step and radius to Tensors
        step = torch.tensor(step, dtype=torch.float32, device=self.device)
        radius = torch.tensor(radius, dtype=torch.float32, device=self.device)

        # Convert degrees to radians
        radians = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32, device=self.device))

        # tx and ty relationships
        tx = step * radius * torch.cos(radians)
        ty = step * radius * torch.sin(radians)

        return -tx, -ty, -1


    def find_alpha_beta(self, vp_x, vp_y, cx, cy, f):
        # Convert inputs to Tensor
        vp_x = torch.tensor(vp_x, dtype=torch.float32, device=self.device)
        vp_y = torch.tensor(vp_y, dtype=torch.float32, device=self.device)
        cx = torch.tensor(cx, dtype=torch.float32, device=self.device)
        cy = torch.tensor(cy, dtype=torch.float32, device=self.device)
        f = torch.tensor(f, dtype=torch.float32, device=self.device)
        
        beta = torch.atan((vp_x - cx) / f)
        alpha = torch.atan((cy - vp_y) / f)
        return alpha, beta


    def run(self, img, VP_parameter):
        vp_x, vp_y, degrees, step = VP_parameter
        alpha, beta = self.find_alpha_beta(vp_x, vp_y, self.cx, self.cy, self.f)
        tx, ty, tz = self.find_tx_ty(step, self.radius, degrees)
        
        # Project 3D points to 2D
        points_2d = self.project_points_to_2d(self.points_3D, self.f, self.f, self.cx, self.cy, alpha, beta, 0, tx, ty, tz)

        # Convert the image to a tensor
        img_tensor = torch.tensor(img, dtype=torch.uint8, device=self.device)

        if len(img_tensor.shape) == 3:
            height, width, _ = img_tensor.shape
            valid_indices = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < width) & (0 <= points_2d[:, 1]) & (points_2d[:, 1] < height)

            points_rgb = torch.zeros((points_2d.shape[0], 3), dtype=torch.uint8, device=self.device)
            points_rgb[valid_indices] = img_tensor[points_2d[valid_indices, 1], points_2d[valid_indices, 0]]
            points_rgb = points_rgb.view(self.L_num, self.R_num, 3)
            
        else:
            height, width = img_tensor.shape
            valid_indices = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < width) & (0 <= points_2d[:, 1]) & (points_2d[:, 1] < height)

            points_rgb = torch.zeros((points_2d.shape[0]), dtype=torch.uint8, device=self.device)
            points_rgb[valid_indices] = img_tensor[points_2d[valid_indices, 1], points_2d[valid_indices, 0]]
            points_rgb = points_rgb.view(self.L_num, self.R_num, 3)

        return points_rgb.cpu().numpy()

