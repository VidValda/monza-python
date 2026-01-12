"""
Monza Ball Physics Simulation with Fuzzy Control

This module simulates a ball rolling on a track with fuzzy logic control
for platform tilt adjustment. The simulation includes physics modeling,
fuzzy control, and real-time visualization.
"""

import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONSTANTS
# ============================================================================

# Setpoints for each level and floor (X, Y coordinates)
# Format: [level_index][floor_index][x_or_y]
# Level indices: 0-3 (Python 0-based, corresponds to MATLAB levels 1-4)
# Floor indices: 0-6 (Python 0-based, corresponds to MATLAB floors 2-8)
SETPOINTS = np.array([
    # Level 1 (index 0)
    [[0, 0, 0, 0, 0, 0, -0.02554],
     [0.11429, 0.06857, 0.02286, -0.02286, -0.06857, -0.11429, -0.16035]],
    # Level 2 (index 1)
    [[0.06211, -0.04758, 0.04969, -0.04847, 0.04406, -0.05409, -0.02554],
     [0.11223, 0.06799, 0.02154, -0.02411, -0.06961, -0.11585, -0.16035]],
    # Level 3 (index 2)
    [[0.12394, -0.09503, 0.09924, -0.0968, 0.08803, -0.108, -0.02554],
     [0.10605, 0.06374, 0.01759, -0.02787, -0.07271, -0.12053, -0.16035]],
    # Level 4 (index 3)
    [[0.12394, 0.14224, 0.14851, -0.14488, 0.1318, -0.108, -0.02554],
     [0.10605, 0.05771, 0.01102, -0.03412, -0.07789, -0.12053, -0.16035]]
])

# Simulation constants
NUM_LEVELS = 4
NUM_FLOORS = 7
DEFAULT_SAMPLING_TIME = 0.033  # seconds
DEFAULT_SIMULATION_DURATION = 8.5  # seconds
EPSILON = 1e-9  # Small value to avoid division by zero
MIN_VELOCITY_FOR_STICK = 0.5  # m/s threshold for ball to stick to floor
PENETRATION_FIX_OFFSET = 0.001  # meters
TRACE_LENGTH = 200  # Number of points to show in ball trajectory
PHASE_TRACE_LENGTH = 500  # Number of points to show in phase plot
DEFUZZIFICATION_RESOLUTION = 1000  # Points for centroid calculation

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class PhysicsParams:
    """Physics simulation parameters."""
    dt: float = 0.01
    gravity: float = 9.81
    friction: float = 0.0223
    restitution: float = 0.1  # Bounciness coefficient
    sub_steps: int = 10       # Physics accuracy (number of sub-steps per frame)

@dataclass
class SimulationConfig:
    """Main simulation configuration."""
    diff_path: str = 'dificultad1.json'
    circ_path: str = 'circulos.json'
    duration_steps: int = 2000
    max_tilt: float = 45.0    # Maximum tilt angle in degrees
    max_omega: float = 3    # Maximum angular velocity in rad/s
    nivel: int = 1            # Current level (1-4, MATLAB 1-based)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def rotate_vector(x: float, y: float, angle: float) -> Tuple[float, float]:
    """
    Rotate a vector from local to global coordinates.
    
    Args:
        x: X component in local coordinates
        y: Y component in local coordinates
        angle: Rotation angle in radians
        
    Returns:
        Tuple of (global_x, global_y) coordinates
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle

def inverse_rotate_vector(x: float, y: float, angle: float) -> Tuple[float, float]:
    """
    Rotate a vector from global to local coordinates (inverse rotation).
    
    Args:
        x: X component in global coordinates
        y: Y component in global coordinates
        angle: Rotation angle in radians
        
    Returns:
        Tuple of (local_x, local_y) coordinates
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return x * cos_angle + y * sin_angle, -x * sin_angle + y * cos_angle

def get_setpoint(level: int, floor: int, coordinate: int = 0) -> float:
    """
    Get setpoint coordinate for a given level and floor.
    
    Args:
        level: Level number (1-4, MATLAB 1-based)
        floor: Floor number (2-8, MATLAB 1-based, converted to 0-6)
        coordinate: 0 for X, 1 for Y
        
    Returns:
        Setpoint coordinate value
    """
    level_idx = max(0, min(level - 1, NUM_LEVELS - 1))
    floor_idx = max(0, min(floor - 2, NUM_FLOORS - 1))
    return SETPOINTS[level_idx, coordinate, floor_idx]

def get_all_setpoints_for_level(level: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get all setpoints (X and Y) for a given level.
    
    Args:
        level: Level number (1-4, MATLAB 1-based)
        
    Returns:
        Tuple of (x_coordinates, y_coordinates) arrays
    """
    level_idx = max(0, min(level - 1, NUM_LEVELS - 1))
    return SETPOINTS[level_idx, 0, :], SETPOINTS[level_idx, 1, :]

# ============================================================================
# LEVEL CLASS
# ============================================================================

class Level:
    """
    Represents a game level with floor segments and visual elements.
    
    Loads level geometry from JSON files and provides interpolation
    functions for floor surfaces.
    """
    
    MAX_FLOOR_SEGMENTS = 20
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize level from configuration.
        
        Args:
            config: Simulation configuration containing file paths
        """
        self.floors: Dict[int, Dict] = {}
        self.visual_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        self._load_level_data(config.diff_path, config.circ_path)

    def _load_level_data(self, difficulty_path: str, circles_path: str) -> None:
        """
        Load level geometry from JSON files.
        
        Args:
            difficulty_path: Path to difficulty/level geometry file
            circles_path: Path to circles geometry file
        """
        try:
            with open(difficulty_path, 'r') as f:
                difficulty_data = json.load(f)
            with open(circles_path, 'r') as f:
                circles_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: Could not load level files: {e}")
            return

        self._process_floor_segments(difficulty_data)
        self._process_line_segments(difficulty_data)
        self._process_circle_segments(circles_data)

    def _process_floor_segments(self, data: Dict) -> None:
        """Process floor segments (xp, yp, xp0, yp0, etc.)."""
        for i in range(self.MAX_FLOOR_SEGMENTS):
            x_key = f'xp{i}' if i > 0 else 'xp'
            y_key = f'yp{i}' if i > 0 else 'yp'
            
            if x_key in data and y_key in data:
                self._process_floor_segment(i, data[x_key], data[y_key])

    def _process_line_segments(self, data: Dict) -> None:
        """Process visual line segments (xl1, yl1, xl2, yl2, etc.)."""
        i = 1
        while True:
            x_key, y_key = f'xl{i}', f'yl{i}'
            if x_key not in data:
                break
            x_data = np.array(data[x_key]).flatten()
            y_data = np.array(data[y_key]).flatten()
            self.visual_segments.append((x_data, y_data))
            i += 1

    def _process_circle_segments(self, data: Dict) -> None:
        """Process circle segments from circles data."""
        r_keys = sorted(
            [k for k in data if k.startswith('r')],
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
        )
        for i in range(0, len(r_keys), 2):
            if i + 1 < len(r_keys):
                x_data = np.array(data[r_keys[i]]).flatten()
                y_data = np.array(data[r_keys[i + 1]]).flatten()
                self.visual_segments.append((x_data, y_data))

    def _process_floor_segment(self, index: int, x_raw: List[float], y_raw: List[float]) -> None:
        """
        Process a single floor segment and create interpolation functions.
        
        Args:
            index: Floor segment index
            x_raw: Raw X coordinates
            y_raw: Raw Y coordinates
        """
        x = np.array(x_raw).flatten()
        y = np.array(y_raw).flatten()
        
        # Sort by X to ensure interpolation works correctly
        sort_indices = np.argsort(x)
        x, y = x[sort_indices], y[sort_indices]
        
        # Calculate slope derivatives
        dx = np.gradient(x)
        dx[dx == 0] = EPSILON  # Avoid division by zero
        slope = np.gradient(y) / dx

        self.floors[index] = {
            'func': interp1d(x, y, kind='cubic', fill_value="extrapolate"),
            'slope': interp1d(x, slope, kind='linear', fill_value="extrapolate"),
            'min': float(x[0]),
            'max': float(x[-1])
        }
        self.visual_segments.append((x, y))

# ============================================================================
# PHYSICS BALL CLASS
# ============================================================================

class PhysicsBall:
    """
    Physics simulation of a ball rolling on a track.
    
    Handles rolling physics, falling physics, and collision detection
    with floor segments.
    """
    
    INITIAL_LOCAL_X = -0.1
    
    def __init__(self, level: Level, start_idx: int = 0):
        """
        Initialize ball physics state.
        
        Args:
            level: Level object containing floor geometry
            start_idx: Starting floor index (default: 0)
        """
        self.level = level
        self.params = PhysicsParams()
        
        # State tracking
        self.current_floor_idx = start_idx
        self.previous_floor_idx = start_idx  # For collision detection
        self.state = "ROLLING"
        
        # Local coordinates (relative to track)
        self.local_x = self.INITIAL_LOCAL_X
        self.local_y = 0.0
        self.local_v = 0.0
        
        # Global coordinates
        self.global_x = 0.0
        self.global_y = 0.0
        self.global_vx = 0.0
        self.global_vy = 0.0

        # Initialize position on starting floor
        self._initialize_position()

    def _initialize_position(self) -> None:
        """Initialize ball position on the starting floor."""
        if self.current_floor_idx in self.level.floors:
            floor_func = self.level.floors[self.current_floor_idx]['func']
            self.local_y = float(floor_func(self.local_x))
        else:
            self.current_floor_idx = -1
            self.local_y = 0.0
            self.state = "FALLING"

    def update(self, angle: float, angular_velocity: float) -> Tuple[float, float]:
        """
        Update ball physics for one time step.
        
        Args:
            angle: Current platform angle in radians
            angular_velocity: Platform angular velocity in rad/s
            
        Returns:
            Tuple of (global_x, global_y) coordinates
        """
        dt_step = self.params.dt / self.params.sub_steps
        
        for _ in range(self.params.sub_steps):
            if self.state == "ROLLING":
                self._handle_rolling(dt_step, angle, angular_velocity)
            elif self.state == "FALLING":
                self._handle_falling(dt_step, angle)
                
        return self.global_x, self.global_y

    def _handle_rolling(self, dt: float, angle: float, angular_velocity: float) -> None:
        """
        Handle ball rolling physics on a floor segment.
        
        Args:
            dt: Time step
            angle: Platform angle in radians
            angular_velocity: Platform angular velocity in rad/s
        """
        if self.current_floor_idx not in self.level.floors:
            self.state = "FALLING"
            return

        floor = self.level.floors[self.current_floor_idx]
        slope_val = float(floor['slope'](self.local_x))
        floor_angle = np.arctan(slope_val)
        
        # Physics: Gravity component along slope + friction
        total_angle = floor_angle + angle
        gravity_component = -self.params.gravity * np.sin(total_angle)
        friction_component = -self.params.friction * self.local_v
        acceleration = gravity_component + friction_component
        
        # Update velocity and position
        self.local_v += acceleration * dt
        self.local_x += self.local_v * dt
        self.local_y = float(floor['func'](self.local_x))
        
        # Convert position to global coordinates
        self.global_x, self.global_y = rotate_vector(self.local_x, self.local_y, angle)
        
        # Update global velocity from local velocity
        # Local velocity components along the track
        local_vx = self.local_v
        local_vy = slope_val * self.local_v
        
        # Rotate local velocity to global coordinates
        global_vx_relative, global_vy_relative = rotate_vector(local_vx, local_vy, angle)
        
        # Add tangential velocity from platform rotation
        tangential_vx = -angular_velocity * self.global_y
        tangential_vy = angular_velocity * self.global_x
        
        self.global_vx = global_vx_relative + tangential_vx
        self.global_vy = global_vy_relative + tangential_vy
        
        # Check if ball rolled off the edge
        if self.local_x < floor['min'] or self.local_x > floor['max']:
            self._transition_to_falling(slope_val, angle, angular_velocity)

    def _transition_to_falling(self, slope: float, angle: float, angular_velocity: float) -> None:
        """
        Transition ball from rolling to falling state.
        
        Args:
            slope: Slope of the floor at transition point
            angle: Platform angle in radians
            angular_velocity: Platform angular velocity in rad/s
        """
        self.previous_floor_idx = self.current_floor_idx
        self.state = "FALLING"
        
        # Convert local velocity to global velocity
        local_vx = self.local_v
        local_vy = slope * self.local_v
        global_vx_relative, global_vy_relative = rotate_vector(local_vx, local_vy, angle)
        
        # Add tangential velocity from platform rotation
        tangential_vx = -angular_velocity * self.global_y
        tangential_vy = angular_velocity * self.global_x
        
        self.global_vx = global_vx_relative + tangential_vx
        self.global_vy = global_vy_relative + tangential_vy

    def _handle_falling(self, dt: float, angle: float) -> None:
        """
        Handle ball falling physics and collision detection.
        
        Args:
            dt: Time step
            angle: Platform angle in radians
        """
        # Apply gravity
        self.global_vy -= self.params.gravity * dt
        self.global_x += self.global_vx * dt
        self.global_y += self.global_vy * dt
        
        # Convert to local coordinates for collision detection
        local_x_check, local_y_check = inverse_rotate_vector(self.global_x, self.global_y, angle)
        
        # Find best collision (floor most below the ball)
        best_collision = self._find_best_collision(local_x_check, local_y_check, angle)
        
        if best_collision is not None:
            floor_idx, floor, target_y = best_collision
            self._resolve_collision(local_x_check, local_y_check, target_y, floor, angle, floor_idx)

    def _find_best_collision(self, local_x: float, local_y: float, angle: float) -> Optional[Tuple[int, Dict, float]]:
        """
        Find the best floor collision candidate.
        
        Only checks floors ahead of the previous floor to prevent backward progression.
        
        Args:
            local_x: Ball X position in local coordinates
            local_y: Ball Y position in local coordinates
            angle: Platform angle in radians
            
        Returns:
            Tuple of (floor_idx, floor_dict, target_y) or None if no collision
        """
        best_collision = None
        max_target_y = float('-inf')
        
        # Determine which floors to check (only forward progression)
        if self.previous_floor_idx >= 0 and self.previous_floor_idx in self.level.floors:
            floors_to_check = [
                idx for idx in sorted(self.level.floors.keys())
                if idx > self.previous_floor_idx
            ]
        else:
            floors_to_check = sorted(self.level.floors.keys())
        
        for floor_idx in floors_to_check:
            floor = self.level.floors[floor_idx]
            
            # Check if ball is within floor X bounds
            if not (floor['min'] <= local_x <= floor['max']):
                continue
                
            target_y = float(floor['func'](local_x))
            
            # Check if ball is at or below the floor
            if local_y > target_y:
                continue
                
            # Check if ball is moving toward the floor
            slope = float(floor['slope'](local_x))
            floor_angle = np.arctan(slope) + angle
            normal_x = -np.sin(floor_angle)
            normal_y = np.cos(floor_angle)
            velocity_dot_normal = self.global_vx * normal_x + self.global_vy * normal_y
            
            # Only consider collision if moving toward floor and prefer lower floors
            if velocity_dot_normal < 0 and target_y > max_target_y:
                max_target_y = target_y
                best_collision = (floor_idx, floor, target_y)
        
        return best_collision

    def _resolve_collision(self, local_x: float, local_y: float, target_y: float,
                          floor: Dict, angle: float, floor_idx: int) -> None:
        """
        Resolve collision with a floor segment.
        
        Args:
            local_x: Ball X position in local coordinates
            local_y: Ball Y position in local coordinates
            target_y: Target Y position on floor
            floor: Floor dictionary with interpolation functions
            angle: Platform angle in radians
            floor_idx: Index of the floor segment
        """
        slope = float(floor['slope'](local_x))
        floor_angle = np.arctan(slope) + angle
        
        # Calculate normal vector
        normal_x = -np.sin(floor_angle)
        normal_y = np.cos(floor_angle)
        velocity_dot_normal = self.global_vx * normal_x + self.global_vy * normal_y
        
        if velocity_dot_normal < 0:
            # Apply bounce with restitution
            impulse = -(1 + self.params.restitution) * velocity_dot_normal
            self.global_vx += impulse * normal_x
            self.global_vy += impulse * normal_y
            
            # Fix penetration
            penetration_fix = target_y - local_y + PENETRATION_FIX_OFFSET
            self.global_x += normal_x * penetration_fix
            self.global_y += normal_y * penetration_fix

            # Transition to rolling if impact velocity is low
            if abs(velocity_dot_normal) < MIN_VELOCITY_FOR_STICK:
                self._transition_to_rolling(local_x, target_y, floor_angle, floor_idx)

    def _transition_to_rolling(self, local_x: float, local_y: float,
                               floor_angle: float, floor_idx: int) -> None:
        """
        Transition ball from falling to rolling state.
        
        Args:
            local_x: Ball X position in local coordinates
            local_y: Ball Y position in local coordinates
            floor_angle: Angle of the floor surface
            floor_idx: Index of the floor segment
        """
        self.state = "ROLLING"
        self.previous_floor_idx = self.current_floor_idx
        self.current_floor_idx = floor_idx
        self.local_x = local_x
        self.local_y = local_y
        
        # Project global velocity onto local tangent direction
        tangent_x = np.cos(floor_angle)
        tangent_y = np.sin(floor_angle)
        self.local_v = self.global_vx * tangent_x + self.global_vy * tangent_y


# ============================================================================
# FUZZY CONTROLLER CLASS
# ============================================================================

class FuzzyController:
    """
    Fuzzy logic controller for platform tilt adjustment.
    
    Based on MATLAB fuzzy logic system with:
    - Input 1: error (9 membership functions)
    - Input 2: velocidad (5 membership functions)
    - Output: inclinacion (9 membership functions)
    
    Uses triangular membership functions, min-max inference, and centroid defuzzification.
    """
    def __init__(self):
        # Input 1: error - 9 membership functions
        self.error_labels = ['grandeNeg', 'medioNeg', 'pequeñoNeg', 'muyPequeñoNeg', 'cero',
                             'muyPequeñoPos', 'pequeñoPos', 'medioPos', 'grandePos']
        self.error_mfs = {
            'grandeNeg': [-0.278, -0.23, -0.182],
            'medioNeg': [-0.22, -0.173, -0.125],
            'pequeñoNeg': [-0.163, -0.115, -0.06],
            'muyPequeñoNeg': [-0.08, -0.04, -0.005],
            'cero': [-0.005, 0, 0.005],
            'muyPequeñoPos': [0.005, 0.04, 0.08],
            'pequeñoPos': [0.06, 0.115, 0.163],
            'medioPos': [0.125, 0.173, 0.22],
            'grandePos': [0.182, 0.23, 0.278]
        }
        # Update error_range to include all membership functions
        all_values = [val for mf in self.error_mfs.values() for val in mf]
        self.error_range = [min(all_values), max(all_values)]
        
        # Input 2: velocidad - 5 membership functions
        # Edge membership functions use trapinf (extend to infinity)
        self.velocidad_labels = ['rapidaNeg', 'lentaNeg', 'cero', 'lentaPos', 'rapidaPos']
    
        # Base membership function values (before scaling)
        velocidad_mfs_base = {
            'rapidaNeg': [-1.5, -0.5, -0.3],  # trapinf: extends to -inf on left
            'lentaNeg': [-0.5, -0.3, 0],
            'cero': [-0.3, 0, 0.3],
            'lentaPos': [0, 0.3, 0.5],
            'rapidaPos': [0.3, 0.5, 1.5]      # trapinf: extends to +inf on right
        }
        # Apply scale factor
        VELOCIDAD_MF_SCALE = 1.1
        self.velocidad_mfs = {
            label: [v * VELOCIDAD_MF_SCALE for v in values]
            for label, values in velocidad_mfs_base.items()
        }
        # Update velocidad_range to include all membership functions
        all_values = [val for mf in self.velocidad_mfs.values() for val in mf]
        self.velocidad_range = [min(all_values), max(all_values)]
        
        # Output: inclinacion - 9 membership functions
        self.inclinacion_labels = ['giraMuchoNeg', 'giraMedioNeg', 'giraPocoNeg', 'giraMuyPocoNeg', 'cero',
                                   'giraMuyPocoPos', 'giraPocoPos', 'giraMedioPos', 'giraMuchoPos']
        self.inclinacion_range = [-0.4, 0.4]
        self.inclinacion_mfs = {
            'giraMuchoNeg': [-0.483333333333333, -0.4, -0.316666666666667],
            'giraMedioNeg': [-0.383333333333333, -0.3, -0.216666666666667],
            'giraPocoNeg': [-0.283333333333333, -0.2, -0.116666666666667],
            'giraMuyPocoNeg': [-0.183333333333333, -0.1, -0.0166666666666667],
            'cero': [-0.0833333333333334, -2.77555756156289e-17, 0.0833333333333333],
            'giraMuyPocoPos': [0.0166666666666666, 0.1, 0.183333333333333],
            'giraPocoPos': [0.116666666666667, 0.2, 0.283333333333333],
            'giraMedioPos': [0.216666666666667, 0.3, 0.383333333333333],
            'giraMuchoPos': [0.316666666666667, 0.4, 0.483333333333333]
        }
        
        # Rules: format is (error_mf_idx, velocidad_mf_idx, inclinacion_mf_idx)
        # MATLAB uses 1-based indexing, we convert to 0-based
        # Rule format: "1 3, 1 (1) : 1" means error=MF1, velocidad=MF3, output=MF1
        self.rules = [
            (0, 2, 0),   # 1 3, 1 (1) : 1 -> error=grandeNeg, velocidad=cero, output=giraMuchoNeg
            (1, 2, 1),   # 2 3, 2 (1) : 1 -> error=medioNeg, velocidad=cero, output=giraMedioNeg
            (2, 2, 2),   # 3 3, 3 (1) : 1 -> error=pequeñoNeg, velocidad=cero, output=giraPocoNeg
            (3, 2, 3),   # 4 3, 4 (1) : 1 -> error=muyPequeñoNeg, velocidad=cero, output=giraMuyPocoNeg
            (5, 2, 5),   # 6 3, 6 (1) : 1 -> error=muyPequeñoPos, velocidad=cero, output=giraMuyPocoPos
            (6, 2, 6),   # 7 3, 7 (1) : 1 -> error=pequeñoPos, velocidad=cero, output=giraPocoPos
            (7, 2, 7),   # 8 3, 8 (1) : 1 -> error=medioPos, velocidad=cero, output=giraMedioPos
            (8, 2, 8),   # 9 3, 9 (1) : 1 -> error=grandePos, velocidad=cero, output=giraMuchoPos
            (0, 1, 0),   # 1 2, 1 (1) : 1 -> error=grandeNeg, velocidad=lentaNeg, output=giraMuchoNeg
            (1, 1, 1),   # 2 2, 2 (1) : 1 -> error=medioNeg, velocidad=lentaNeg, output=giraMedioNeg
            (2, 1, 2),   # 3 2, 3 (1) : 1 -> error=pequeñoNeg, velocidad=lentaNeg, output=giraPocoNeg
            (6, 3, 6),   # 7 4, 7 (1) : 1 -> error=pequeñoPos, velocidad=lentaPos, output=giraPocoPos
            (7, 3, 7),   # 8 4, 8 (1) : 1 -> error=medioPos, velocidad=lentaPos, output=giraMedioPos
            (8, 3, 8),   # 9 4, 9 (1) : 1 -> error=grandePos, velocidad=lentaPos, output=giraMuchoPos
            (3, 0, 3),   # 4 1, 4 (1) : 1 -> error=muyPequeñoNeg, velocidad=rapidaNeg, output=giraMuyPocoNeg
            (5, 4, 5),   # 6 5, 6 (1) : 1 -> error=muyPequeñoPos, velocidad=rapidaPos, output=giraMuyPocoPos
            (2, 0, 2),   # 3 1, 3 (1) : 1 -> error=pequeñoNeg, velocidad=rapidaNeg, output=giraPocoNeg
            (6, 4, 6),   # 7 5, 7 (1) : 1 -> error=pequeñoPos, velocidad=rapidaPos, output=giraPocoPos
            (4, 4, 3),   # 5 5, 4 (1) : 1 -> error=cero, velocidad=rapidaPos, output=giraMuyPocoNeg
            (4, 0, 5),   # 5 1, 6 (1) : 1 -> error=cero, velocidad=rapidaNeg, output=giraMuyPocoPos
        ]

    def _trimf(self, x: float, abc: Tuple[float, float, float]) -> float:
        """
        Triangular membership function.
        
        Args:
            x: Input value
            abc: Tuple of (a, b, c) where a is left base, b is peak, c is right base
            
        Returns:
            Membership value (0.0 to 1.0)
        """
        a, b, c = abc
        if x <= a or x >= c:
            return 0.0
        if x < b:
            return (x - a) / (b - a + EPSILON)
        else:
            return (c - x) / (c - b + EPSILON)

    def _trapinf(self, x: float, abc: Tuple[float, float, float], is_left_edge: bool) -> float:
        """
        Trapezoidal infinity membership function for edge membership functions.
        Extends to infinity on one side.
        
        Args:
            x: Input value
            abc: Tuple of (a, b, c) where a is left base, b is peak, c is right base
            is_left_edge: True for left edge (extends to -inf), False for right edge (extends to +inf)
            
        Returns:
            Membership value (0.0 to 1.0)
        """
        a, b, c = abc
        if is_left_edge:
            # Left edge: membership = 1.0 for x <= b, then decreases to 0 at c
            if x <= b:
                return 1.0
            elif x >= c:
                return 0.0
            else:
                return (c - x) / (c - b + EPSILON)
        else:
            # Right edge: membership = 0 for x <= a, then increases to 1.0 at b
            if x <= a:
                return 0.0
            elif x >= b:
                return 1.0
            else:
                return (x - a) / (b - a + EPSILON)

    def _fuzzify_error(self, error_val: float) -> Dict[str, float]:
        """
        Fuzzify error input.
        
        Args:
            error_val: Error value to fuzzify
            
        Returns:
            Dictionary mapping error membership function labels to membership values
        """
        memberships = {}
        for label in self.error_labels:
            memberships[label] = self._trimf(error_val, self.error_mfs[label])
        return memberships

    def _fuzzify_velocidad(self, velocidad_val: float) -> Dict[str, float]:
        """
        Fuzzify velocidad input.
        Edge membership functions (rapidaNeg, rapidaPos) use trapinf.
        
        Args:
            velocidad_val: Velocity value to fuzzify
            
        Returns:
            Dictionary mapping velocidad membership function labels to membership values
        """
        memberships = {}
        for label in self.velocidad_labels:
            if label == 'rapidaNeg':
                # Left edge: extends to -inf
                memberships[label] = self._trapinf(velocidad_val, self.velocidad_mfs[label], is_left_edge=True)
            elif label == 'rapidaPos':
                # Right edge: extends to +inf
                memberships[label] = self._trapinf(velocidad_val, self.velocidad_mfs[label], is_left_edge=False)
            else:
                # Middle membership functions use triangular
                memberships[label] = self._trimf(velocidad_val, self.velocidad_mfs[label])
        return memberships

    def _centroid_defuzzify(self, aggregated_output: Dict[str, float]) -> float:
        """
        Centroid defuzzification method.
        
        Args:
            aggregated_output: Dictionary mapping output labels to their aggregated membership values
            
        Returns:
            Defuzzified output value (centroid)
        """
        numerator = 0.0
        denominator = 0.0
        
        # Use fine resolution for centroid calculation
        output_range = np.linspace(
            self.inclinacion_range[0],
            self.inclinacion_range[1],
            DEFUZZIFICATION_RESOLUTION
        )
        
        for x in output_range:
            # Calculate membership value at x for each output MF
            mu_x = 0.0
            for label in self.inclinacion_labels:
                mu_mf = self._trimf(x, self.inclinacion_mfs[label])
                # Aggregate using max (as per AggMethod='max')
                mu_x = max(mu_x, min(mu_mf, aggregated_output.get(label, 0.0)))
            
            numerator += x * mu_x
            denominator += mu_x
        
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def compute(self, error: float, velocidad: float) -> Tuple[float, Dict]:
        """
        Compute fuzzy controller output.
        
        Args:
            error: Position error value
            velocidad: Velocity value
            
        Returns:
            Tuple of (output, debug_info) where:
            - output: Defuzzified output (inclinacion in radians)
            - debug_info: Dictionary containing fuzzification results, active rules, etc.
        """
        # Clamp inputs to ranges
        error = np.clip(error, self.error_range[0], self.error_range[1])
        velocidad = np.clip(velocidad, self.velocidad_range[0], self.velocidad_range[1])
        
        # 1. Fuzzification
        m_error = self._fuzzify_error(error)
        m_velocidad = self._fuzzify_velocidad(velocidad)
        
        # 2. Rule evaluation and aggregation
        # Initialize aggregated output (using max aggregation)
        aggregated_output = {label: 0.0 for label in self.inclinacion_labels}
        active_rules = []
        
        for error_mf_idx, velocidad_mf_idx, inclinacion_mf_idx in self.rules:
            # Get membership values
            error_label = self.error_labels[error_mf_idx]
            velocidad_label = self.velocidad_labels[velocidad_mf_idx]
            inclinacion_label = self.inclinacion_labels[inclinacion_mf_idx]
            
            # AND method: min
            rule_strength = min(m_error[error_label], m_velocidad[velocidad_label])
            
            if rule_strength > 0:
                # Implication method: min (clip output MF by rule strength)
                # Aggregation method: max (take maximum of all rule outputs)
                aggregated_output[inclinacion_label] = max(
                    aggregated_output[inclinacion_label],
                    rule_strength  # min implication
                )
                
                # Debug info
                active_rules.append({
                    'error_mf': error_label,
                    'velocidad_mf': velocidad_label,
                    'inclinacion_mf': inclinacion_label,
                    'strength': rule_strength,
                    'indices': (velocidad_mf_idx, error_mf_idx)
                })
        
        # 3. Defuzzification: centroid
        output = self._centroid_defuzzify(aggregated_output)
        
        return output, {
            'm_error': m_error,
            'm_velocidad': m_velocidad,
            'rules': active_rules,
            'output': output,
            'aggregated_output': aggregated_output
        }
        
# ============================================================================
# DASHBOARD CLASS
# ============================================================================

class Dashboard:
    """
    Real-time visualization dashboard for the simulation.
    
    Displays track view, fuzzy control visualization, state information,
    and various plots showing system behavior over time.
    """
    
    def __init__(self, level: Level, fuzzy: FuzzyController, history: Dict, config: SimulationConfig):
        """
        Initialize dashboard with simulation data.
        
        Args:
            level: Level object with track geometry
            fuzzy: Fuzzy controller instance
            history: Dictionary containing simulation history
            config: Simulation configuration
        """
        self.level = level
        self.fuzzy = fuzzy
        self.history = history
        self.config = config
        # Larger figure with better proportions
        self.fig = plt.figure(figsize=(24, 16))
        # Better grid: 8 rows x 5 columns with improved spacing
        self.gs = self.fig.add_gridspec(6, 5, hspace=1, wspace=0.4, 
                                        left=0.06, right=0.97, top=0.95, bottom=0.05)
        self.artists = []
        
        self._setup_track_view()
        self._setup_heatmap()
        self._setup_defuzz_view()
        self._setup_state_info()
        self._setup_setpoint_plot()
        self._setup_velocity_plot()
        self._setup_control_output_plot()
        self._setup_platform_angle_plot()
        self._setup_phase_plot()
        self._setup_fuzz_pos()
        self._setup_fuzz_vel()
        self._setup_output_mf_view()

    def _setup_track_view(self):
        ax = self.fig.add_subplot(self.gs[0:3, 0:2])
        ax.set_title("Simulation Track", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.3, 0.3); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X Position (m)", fontsize=11)
        ax.set_ylabel("Y Position (m)", fontsize=11)
        
        # Plot all setpoints for current level
        setpoints_x, setpoints_y = get_all_setpoints_for_level(self.config.nivel)
        self.setpoint_markers, = ax.plot(setpoints_x, setpoints_y, 'g*', markersize=12, 
                                         label='Setpoints', zorder=5, alpha=0.7)
        self.current_setpoint_marker, = ax.plot([], [], 'gX', markersize=15, 
                                                label='Current Target', zorder=6, markeredgewidth=2)
        
        self.track_lines = [ax.plot([], [], 'k-', lw=1.5)[0] for _ in self.level.visual_segments]
        self.ball_dot, = ax.plot([], [], 'ro', markersize=10, zorder=10, label='Ball')
        self.ball_trace, = ax.plot([], [], 'r-', lw=1, alpha=0.6, label='Trajectory')
        #ax.legend(loc='upper right', fontsize=8)

    def _setup_heatmap(self):
        ax = self.fig.add_subplot(self.gs[0:1, 2:3])
        ax.set_title("Active Rules Matrix", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Error", fontsize=10); ax.set_ylabel("Velocidad", fontsize=10)
        # Error has 9 MFs, Velocidad has 5 MFs
        ax.set_xticks(range(9)); ax.set_xticklabels([l[:4] for l in self.fuzzy.error_labels], rotation=45, fontsize=8)
        ax.set_yticks(range(5)); ax.set_yticklabels([l[:4] for l in self.fuzzy.velocidad_labels], fontsize=8)
        self.heatmap_img = ax.imshow(np.zeros((5, 9)), cmap='Reds', vmin=0, vmax=1, origin='lower', aspect='auto')
        plt.colorbar(self.heatmap_img, ax=ax, fraction=0.046, pad=0.04)

    def _setup_defuzz_view(self):
        ax = self.fig.add_subplot(self.gs[2:3, 2:3])
        ax.set_title("Defuzzification", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Error", fontsize=9)
        # Get centers of output MFs
        output_centers = [self.fuzzy.inclinacion_mfs[label][1] for label in self.fuzzy.inclinacion_labels]
        for i, (label, center) in enumerate(zip(self.fuzzy.inclinacion_labels, output_centers)):
            ax.axvline(center, color='gray', linestyle=':', alpha=0.5)
            ax.text(center, 1.02, label[:4], ha='center', fontsize=7, rotation=45)
        self.out_bars = ax.bar(output_centers, [0]*9, width=0.02, color='blue', alpha=0.6)
        self.defuzz_out_line = ax.axvline(0, color='red', lw=2)
        self.output_centers = output_centers

    def _setup_fuzz_plot(self, gs_pos, title, mf_dict, mf_labels, mf_range, color):
        ax = self.fig.add_subplot(gs_pos)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(mf_range[0] * 1.2, mf_range[1] * 1.2)
        ax.set_xlabel(title.split(':')[-1].strip(), fontsize=9)
        ax.set_ylabel("Membership", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Check if this is velocity membership functions (for trapinf edges)
        is_velocidad = (mf_dict is self.fuzzy.velocidad_mfs)
        
        # Draw static MF triangles (or trapinf for edges)
        x_static = np.linspace(mf_range[0] * 1.2, mf_range[1] * 1.2, 200)
        for label in mf_labels:
            abc = mf_dict[label]
            if is_velocidad and label == 'rapidaNeg':
                # Left edge: extends to -inf
                y = [self.fuzzy._trapinf(xi, abc, is_left_edge=True) for xi in x_static]
            elif is_velocidad and label == 'rapidaPos':
                # Right edge: extends to +inf
                y = [self.fuzzy._trapinf(xi, abc, is_left_edge=False) for xi in x_static]
            else:
                # Regular triangular membership function
                y = [self.fuzzy._trimf(xi, abc) for xi in x_static]
            ax.plot(x_static, y, 'k-', lw=0.5, alpha=0.5)
            ax.fill_between(x_static, 0, y, alpha=0.05, color=color)
        
        line = ax.axvline(0, color='red', lw=1.5)
        dots, = ax.plot([], [], f'{color[0]}o', markersize=4)
        return line, dots

    def _setup_setpoint_plot(self):
        ax = self.fig.add_subplot(self.gs[3:4, 0:3])
        ax.set_title("Position Tracking", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Position (m)", fontsize=10)
        ax.grid(True, alpha=0.3)
        self.setpoint_line, = ax.plot([], [], 'g--', lw=2, label='Setpoint', alpha=0.8, marker='o', markersize=3, markevery=10)
        self.position_line, = ax.plot([], [], 'b-', lw=1.5, label='Position', alpha=0.8)
        self.error_line, = ax.plot([], [], 'r-', lw=1.2, label='Error', alpha=0.7)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(0, len(self.history['x']))
        self.setpoint_ax = ax

    def _setup_velocity_plot(self):
        ax = self.fig.add_subplot(self.gs[4:5, 0:1])
        ax.set_title("Velocity", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Velocity (m/s)", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.velocity_line, = ax.plot([], [], 'm-', lw=1.5, label='Global Vx', alpha=0.8)
        self.local_velocity_line, = ax.plot([], [], 'c--', lw=1.2, label='Local V', alpha=0.7)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, len(self.history['x']))
        self.velocity_ax = ax

    def _setup_control_output_plot(self):
        ax = self.fig.add_subplot(self.gs[4:5, 1:2])
        ax.set_title("Control Output", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Inclinación (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.control_output_line, = ax.plot([], [], 'orange', lw=1.5, label='Fuzzy Output', alpha=0.8)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, len(self.history['x']))
        self.control_output_ax = ax

    def _setup_platform_angle_plot(self):
        ax = self.fig.add_subplot(self.gs[4:5, 2:3])
        ax.set_title("Platform Angle", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Angle (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.angle_line, = ax.plot([], [], 'purple', lw=1.5, label='Platform Angle', alpha=0.8)
        self.angle_deg_line, = ax.plot([], [], 'brown', lw=1.2, linestyle='--', label='Angle (deg)', alpha=0.7)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, len(self.history['x']))
        self.angle_ax = ax

    def _setup_phase_plot(self):
        ax = self.fig.add_subplot(self.gs[4:5, 3:4])
        ax.set_title("Phase Plot", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Error (m)", fontsize=9)
        ax.set_ylabel("Velocity (m/s)", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.phase_line, = ax.plot([], [], 'b-', lw=1, alpha=0.6, label='Trajectory')
        self.phase_dot, = ax.plot([], [], 'ro', markersize=6, label='Current', zorder=10)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.legend(loc='best', fontsize=8)
        self.phase_ax = ax

    def _setup_state_info(self):
        ax = self.fig.add_subplot(self.gs[1:2, 3:5])
        ax.axis('off')
        ax.set_title("System State", fontsize=12, fontweight='bold', pad=-30, loc='left')
        self.state_text = ax.text(0.1, 0.9, '', transform=ax.transAxes, fontsize=10,
                                  verticalalignment='top', family='monospace',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=8))

    def _setup_fuzz_pos(self):
        self.line_p, self.dots_p = self._setup_fuzz_plot(
            self.gs[5:6, 0:1], "Fuzzification: Error", 
            self.fuzzy.error_mfs, self.fuzzy.error_labels, 
            self.fuzzy.error_range, 'blue'
        )

    def _setup_fuzz_vel(self):
        self.line_v, self.dots_v = self._setup_fuzz_plot(
            self.gs[5:6, 1:2], "Fuzzification: Velocidad", 
            self.fuzzy.velocidad_mfs, self.fuzzy.velocidad_labels,
            self.fuzzy.velocidad_range, 'green'
        )

    def _setup_output_mf_view(self):
        ax = self.fig.add_subplot(self.gs[5:6, 2:3])
        ax.set_title("Output MFs: Inclinación", fontsize=11, fontweight='bold', pad=6)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(self.fuzzy.inclinacion_range[0] * 1.2, self.fuzzy.inclinacion_range[1] * 1.2)
        ax.set_xlabel("Inclinación (rad)", fontsize=9)
        ax.set_ylabel("Membership", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Draw static output MF triangles
        x_static = np.linspace(self.fuzzy.inclinacion_range[0] * 1.2, 
                              self.fuzzy.inclinacion_range[1] * 1.2, 200)
        for label in self.fuzzy.inclinacion_labels:
            abc = self.fuzzy.inclinacion_mfs[label]
            y = [self.fuzzy._trimf(xi, abc) for xi in x_static]
            ax.plot(x_static, y, 'k-', lw=0.5, alpha=0.5)
            ax.fill_between(x_static, 0, y, alpha=0.05, color='orange')
        
        self.output_line = ax.axvline(0, color='red', lw=2, label='Output')
        ax.legend(loc='upper right', fontsize=8)

    def update_frame(self, frame: int):
        """
        Update all dashboard elements for a given frame.
        
        Args:
            frame: Current frame number
            
        Returns:
            List of artists to update
        """
        # 1. Update Track Geometry
        angle = self.history['angle'][frame]
        for line, (local_x, local_y) in zip(self.track_lines, self.level.visual_segments):
            global_x, global_y = rotate_vector(local_x, local_y, angle)
            line.set_data(global_x, global_y)
        
        self.ball_dot.set_data([self.history['x'][frame]], [self.history['y'][frame]])
        trace_len = min(TRACE_LENGTH, frame + 1)
        start_idx = max(0, frame - trace_len + 1)
        self.ball_trace.set_data(
            self.history['x'][start_idx:frame + 1],
            self.history['y'][start_idx:frame + 1]
        )
        
        # Update current setpoint marker
        if frame < len(self.history['setpoint']):
            current_setpoint_x = self.history['setpoint'][frame]
            setpoints_x, setpoints_y = get_all_setpoints_for_level(self.config.nivel)
            closest_idx = np.argmin(np.abs(setpoints_x - current_setpoint_x))
            self.current_setpoint_marker.set_data(
                [setpoints_x[closest_idx]], [setpoints_y[closest_idx]]
            )

        # 2. Update Setpoint Plot
        time_steps = list(range(frame + 1))
        self.setpoint_line.set_data(time_steps, self.history['setpoint'][:frame + 1])
        self.position_line.set_data(time_steps, self.history['x'][:frame + 1])
        self.error_line.set_data(time_steps, self.history['error'][:frame + 1])
        self._update_axis_limits(self.setpoint_ax, frame, ['x', 'setpoint', 'error'])

        # 3. Update Velocity Plot
        self.velocity_line.set_data(time_steps, self.history['global_vx'][:frame + 1])
        self.local_velocity_line.set_data(time_steps, self.history['lv'][:frame + 1])
        self._update_axis_limits(self.velocity_ax, frame, ['global_vx', 'lv'])

        # 4. Update Control Output Plot
        control_outputs = [
            d['output'] if d else 0.0
            for d in self.history['debug'][:frame + 1]
        ]
        self.control_output_line.set_data(time_steps, control_outputs)
        if control_outputs:
            self._update_axis_limits_from_list(self.control_output_ax, control_outputs)

        # 5. Update Platform Angle Plot
        angles = self.history['angle'][:frame + 1]
        angles_deg = [np.degrees(a) for a in angles]
        self.angle_line.set_data(time_steps, angles)
        self.angle_deg_line.set_data(time_steps, angles_deg)
        if angles:
            self._update_axis_limits_from_list(self.angle_ax, angles)

        # 6. Update Phase Plot
        phase_trace_len = min(PHASE_TRACE_LENGTH, frame + 1)
        phase_start = max(0, frame - phase_trace_len + 1)
        phase_errors = self.history['error'][phase_start:frame + 1]
        phase_vels = self.history['global_vx'][phase_start:frame + 1]
        self.phase_line.set_data(phase_errors, phase_vels)
        if frame < len(self.history['error']):
            self.phase_dot.set_data(
                [self.history['error'][frame]],
                [self.history['global_vx'][frame]]
            )
        if phase_errors and phase_vels:
            self.phase_ax.set_xlim(min(phase_errors) - 0.01, max(phase_errors) + 0.01)
            self.phase_ax.set_ylim(min(phase_vels) - 0.01, max(phase_vels) + 0.01)

        # 7. Update State Info
        current_floor = self.history.get('current_floor', [0] * len(self.history['x']))
        current_floor = current_floor[frame] if frame < len(current_floor) else 0
        state_text = self._format_state_text(frame, current_floor)
        self.state_text.set_text(state_text)

        # 8. Update Debug Visuals
        debug_info = self.history['debug'][frame]
        if not debug_info:
            return self._get_artists()

        error_val = self.history['error'][frame]
        velocidad_val = self.history['global_vx'][frame]
        
        # Fuzzification Lines/Dots
        self.line_p.set_xdata([error_val])
        self.line_v.set_xdata([velocidad_val])
        error_memberships = [debug_info['m_error'][k] for k in self.fuzzy.error_labels]
        velocidad_memberships = [debug_info['m_velocidad'][k] for k in self.fuzzy.velocidad_labels]
        self.dots_p.set_data([error_val] * len(self.fuzzy.error_labels), error_memberships)
        self.dots_v.set_data([velocidad_val] * len(self.fuzzy.velocidad_labels), velocidad_memberships)

        # Heatmap: velocidad (rows) x error (columns)
        grid = np.zeros((5, 9))
        for rule in debug_info['rules']:
            velocidad_idx = self.fuzzy.velocidad_labels.index(rule['velocidad_mf'])
            error_idx = self.fuzzy.error_labels.index(rule['error_mf'])
            grid[velocidad_idx, error_idx] = max(grid[velocidad_idx, error_idx], rule['strength'])
        self.heatmap_img.set_data(grid)

        # Defuzzification Bars
        bar_heights = [0] * 9
        for rule in debug_info['rules']:
            inclinacion_idx = self.fuzzy.inclinacion_labels.index(rule['inclinacion_mf'])
            bar_heights[inclinacion_idx] = max(bar_heights[inclinacion_idx], rule['strength'])
        
        for bar, height in zip(self.out_bars, bar_heights):
            bar.set_height(height)
        self.defuzz_out_line.set_xdata([debug_info['output']])
        self.output_line.set_xdata([debug_info['output']])
        
        return self._get_artists()

    def _update_axis_limits(self, ax, frame: int, keys: List[str]) -> None:
        """Update axis limits based on history data."""
        if frame < 0 or not self.history:
            return
        all_values = []
        for key in keys:
            if key in self.history and len(self.history[key]) > 0:
                all_values.extend(self.history[key][:frame + 1])
        if all_values:
            self._update_axis_limits_from_list(ax, all_values)

    def _update_axis_limits_from_list(self, ax, values: List[float]) -> None:
        """Update axis limits from a list of values."""
        if not values:
            return
        value_min, value_max = min(values), max(values)
        value_range = value_max - value_min
        padding = max(0.1 * value_range, 0.01) if value_range > 0 else 0.1
        ax.set_ylim(value_min - padding, value_max + padding)

    def _format_state_text(self, frame: int, current_floor: int) -> str:
        """Format state information text."""
        state_text = f"Level: {self.config.nivel}\n"
        state_text += f"Floor: {current_floor}\n"
        state_text += f"Position: {self.history['x'][frame]:.4f} m\n"
        state_text += f"Error: {self.history['error'][frame]:.4f} m\n"
        state_text += f"Velocity: {self.history['global_vx'][frame]:.4f} m/s\n"
        state_text += f"Angle: {np.degrees(self.history['angle'][frame]):.2f}°\n"
        if frame < len(self.history['debug']) and self.history['debug'][frame]:
            state_text += f"Control Out: {self.history['debug'][frame]['output']:.4f} rad"
        return state_text

    def _get_artists(self):
        """Get all artists for animation update."""
        return (self.track_lines + [
            self.ball_dot, self.ball_trace, self.setpoint_markers,
            self.current_setpoint_marker, self.setpoint_line,
            self.position_line, self.error_line, self.velocity_line,
            self.local_velocity_line, self.control_output_line,
            self.angle_line, self.angle_deg_line, self.phase_line,
            self.phase_dot, self.line_p, self.dots_p,
            self.line_v, self.dots_v, self.heatmap_img, self.defuzz_out_line,
            self.output_line, self.state_text
        ] + list(self.out_bars))

    def show(self):
        """Display the animation."""
        ani = FuncAnimation(
            self.fig, self.update_frame,
            frames=len(self.history['x']),
            interval=20,
            blit=False
        )
        plt.show()

    def save_video(self, filename: str = 'simulation_video.mp4', fps: int = 30, dpi: int = 100):
        """
        Save the animation as a video file.
        
        Args:
            filename: Output video filename (default: 'simulation_video.mp4')
            fps: Frames per second for the video (default: 30)
            dpi: Dots per inch for video quality (default: 100)
        """
        print(f"Saving animation to {filename}...")
        print(f"Total frames: {len(self.history['x'])}")
        print(f"Video settings: {fps} fps, {dpi} dpi")
        
        # Create animation
        ani = FuncAnimation(
            self.fig, self.update_frame,
            frames=len(self.history['x']),
            interval=1000/fps,  # Convert fps to interval in milliseconds
            blit=False,
            repeat=False
        )
        
        # Try to use FFMpegWriter first (better quality), fall back to PillowWriter if not available
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Monza Simulation'), bitrate=1800)
            ani.save(filename, writer=writer, dpi=dpi)
            print(f"Video saved successfully as {filename}")
        except Exception as e:
            print(f"FFMpegWriter failed: {e}")
            print("Trying PillowWriter (GIF format)...")
            try:
                # PillowWriter creates GIF files
                if not filename.endswith('.gif'):
                    filename = filename.rsplit('.', 1)[0] + '.gif'
                writer = PillowWriter(fps=fps)
                ani.save(filename, writer=writer, dpi=dpi)
                print(f"Animation saved as GIF: {filename}")
            except Exception as e2:
                print(f"Failed to save video: {e2}")
                print("Please install ffmpeg for MP4 support: sudo apt-get install ffmpeg")
                raise

# ============================================================================
# CONTROLLER FUNCTION
# ============================================================================

def calculate_position_error(floor: int, position_x: float, level: int) -> float:
    """
    Calculate position error based on current position, floor, and level.
    
    The error is the difference between current X position and the target
    X position for the current floor and level.
    
    Args:
        floor: Current floor index (2-8, MATLAB 1-based)
        position_x: Current X position in global coordinates
        level: Current level (1-4, MATLAB 1-based)
        
    Returns:
        Position error (position_x - target_x)
    """
    target_x = get_setpoint(level, floor, coordinate=0)
    return position_x - target_x

def main():
    # 1. Setup
    config = SimulationConfig()
    level = Level(config)
    ball = PhysicsBall(level, start_idx=1)
    fuzzy = FuzzyController()
    
    # 2. Simulation Loop
    history = {'x': [], 'y': [], 'angle': [], 'lx': [], 'lv': [], 'global_vx': [], 
               'debug': [], 'setpoint': [], 'error': [], 'current_floor': []}
    current_angle = 0.0
    
    # Simulation parameters for CSV export
    sampling_time = DEFAULT_SAMPLING_TIME
    simulation_duration = DEFAULT_SIMULATION_DURATION
    num_steps = int(simulation_duration / sampling_time)
    
    # Update physics dt to match simulation sampling time
    ball.params.dt = sampling_time
    
    print(f"Simulating for {simulation_duration} seconds with Ts = {sampling_time} s ({num_steps} steps)...")
    for step in range(num_steps):
        # Get current floor index (0-based) and convert to MATLAB 1-based floor
        floor = ball.current_floor_idx + 1 if ball.current_floor_idx >= 0 else 1
        
        # Calculate error using controller function
        error = calculate_position_error(floor, ball.global_x, config.nivel)
        
        # Calculate target position (setpoint) for visualization
        target_x = get_setpoint(config.nivel, floor, coordinate=0)
        
        # Compute fuzzy controller output (error, velocidad)
        # Note: velocidad is the velocity in X direction (global_vx)
        output, debug = fuzzy.compute(error, ball.local_v)
        
        # Convert fuzzy output to target angle (output is already in correct range)
        # The output is inclinacion, which directly maps to tilt angle
        max_tilt_rad = np.radians(config.max_tilt)
        target_angle = np.clip(output, -max_tilt_rad, max_tilt_rad)

        # Kinematics Step: Update platform angle
        angle_diff = target_angle - current_angle
        omega = np.clip(angle_diff / sampling_time, -config.max_omega, config.max_omega)
        new_angle = current_angle + omega * sampling_time
        
        # Physics Step
        gx, gy = ball.update(new_angle, omega)
        current_angle = new_angle
        
        # Record Data
        history['x'].append(gx)
        history['y'].append(gy)
        history['angle'].append(current_angle)
        history['lx'].append(ball.local_x)
        history['lv'].append(ball.local_v)
        history['global_vx'].append(ball.global_vx)
        history['setpoint'].append(target_x)
        history['error'].append(error)
        history['debug'].append(debug)
        history['current_floor'].append(floor)

    # 3. Save x and y data to CSV
    csv_filename = 'simulation_data.csv'
    print(f"Saving simulation data to {csv_filename}...")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'x', 'y'])  # Header
        for i in range(len(history['x'])):
            time = i * sampling_time
            writer.writerow([time, history['x'][i], history['y'][i]])
    print(f"Data saved to {csv_filename}")

    # 4. Visualization
    dashboard = Dashboard(level, fuzzy, history, config)
    
    # Save video
    video_filename = 'simulation_video.mp4'
    dashboard.save_video(video_filename, fps=30, dpi=100)
    
    # Optionally display the animation (comment out if you only want the video)
    # dashboard.show()

if __name__ == "__main__":
    main()