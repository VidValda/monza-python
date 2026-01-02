import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# --- 1. DATA LOADING & PROCESSING ---

def process_json_data(diff_filename, circ_filename):
    """
    Parses the JSON content and creates interpolation functions for physics.
    """
    # NOTE: In a local environment, use open(diff_filename).read()
    # For this demonstration, we assume the content is passed or files exist
    try:
        with open(diff_filename, 'r') as f:
            d_data = json.load(f)
        with open(circ_filename, 'r') as f:
            c_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {diff_filename} or {circ_filename}")
        return None

    sim_data = {
        'floors': {},
        'visuals': []
    }

    # 1. Process Floors (Physics & Visuals)
    # The JSON keys are 'xp1', 'yp1' etc.
    for i in range(1, 8):
        x_key = f'xp{i}'
        y_key = f'yp{i}'
        
        if x_key in d_data and y_key in d_data:
            # Flatten lists (JSON gives [[x,x,x]])
            x_arr = np.array(d_data[x_key]).flatten()
            y_arr = np.array(d_data[y_key]).flatten()
            
            # Sort by x for valid interpolation
            sort_idx = np.argsort(x_arr)
            x_arr = x_arr[sort_idx]
            y_arr = y_arr[sort_idx]

            # Create interpolation function for physics (Height y given x)
            # fill_value="extrapolate" allows us to detect when we fall off
            f_interp = interp1d(x_arr, y_arr, kind='cubic', fill_value="extrapolate")
            
            # Calculate slope (derivative) for physics
            # dy/dx approximation
            dx_arr = np.gradient(x_arr)
            dy_arr = np.gradient(y_arr)
            # Avoid division by zero
            dx_arr[dx_arr == 0] = 1e-9
            slope_arr = dy_arr / dx_arr
            s_interp = interp1d(x_arr, slope_arr, kind='linear', fill_value="extrapolate")

            sim_data['floors'][i] = {
                'x': x_arr,
                'y': y_arr,
                'func_y': f_interp,
                'func_slope': s_interp,
                'x_min': x_arr[0],
                'x_max': x_arr[-1]
            }
            sim_data['visuals'].append((x_arr, y_arr))

    # 2. Process Static Lines (Visuals only)
    # xl1, yl1 ...
    for i in range(1, 10): # Arbitrary range to catch all xl/yl
        xk = f'xl{i}'
        yk = f'yl{i}'
        if xk in d_data and yk in d_data:
            sim_data['visuals'].append((
                np.array(d_data[xk]).flatten(), 
                np.array(d_data[yk]).flatten()
            ))

    # 3. Process Base (xp, yp)
    if 'xp' in d_data and 'yp' in d_data:
        sim_data['visuals'].append((
            np.array(d_data['xp']).flatten(),
            np.array(d_data['yp']).flatten()
        ))

    # 4. Process Circles
    # JSON has r1, r2, r3... Assumed pairs (r1=x, r2=y), (r3=x, r4=y)
    r_keys = sorted([k for k in c_data.keys() if k.startswith('r')], key=lambda x: int(x[1:]))
    for i in range(0, len(r_keys), 2):
        if i+1 < len(r_keys):
            x_c = np.array(c_data[r_keys[i]]).flatten()
            y_c = np.array(c_data[r_keys[i+1]]).flatten()
            sim_data['visuals'].append((x_c, y_c))

    return sim_data

# --- 2. PHYSICS CONSTANTS ---
DT = 0.01        # Time step
G = 9.81         # Gravity
M = 0.01         # Mass
FRICTION = 0.05  # Resistive force coef
ROT_SPEED = 0.5  # Amplitude of rotation
FREQ = 0.5       # Frequency of rotation

# --- 3. TRANSFORMATION MATH ---

def to_global(x_loc, y_loc, angle):
    c, s = np.cos(angle), np.sin(angle)
    x_g = x_loc * c - y_loc * s
    y_g = x_loc * s + y_loc * c
    return x_g, y_g

def to_local(x_g, y_g, angle):
    c, s = np.cos(angle), np.sin(angle)
    x_l = x_g * c + y_g * s
    y_l = -x_g * s + y_g * c
    return x_l, y_l

# --- 4. SIMULATION LOOP ---

def run_simulation(data):
    # Initial State
    current_floor_idx = 1
    floor_data = data['floors'][current_floor_idx]
    
    # Start slightly off-center to ensure movement
    x_local = 0.005 
    y_local = float(floor_data['func_y'](x_local))
    v_local = 0.0 # Velocity along the curve (tangential speed roughly)
    
    # Global State
    x_glob, y_glob = 0, 0
    vx_glob, vy_glob = 0, 0
    
    state = "ROLLING" # ROLLING, FALLING
    
    history = {'x': [], 'y': [], 'angle': [], 'state': []}
    
    # Max simulation time
    steps = 1000 
    
    for step in range(steps):
        t = step * DT
        
        # Calculate System Rotation
        # Angle oscillates
        theta = ROT_SPEED * np.sin(FREQ * t)
        omega = ROT_SPEED * FREQ * np.cos(FREQ * t) # Angular velocity (d(theta)/dt)
        
        # --- STATE: ROLLING ---
        if state == "ROLLING":
            # 1. Get Geometry from Data Interpolation
            try:
                floor_y = float(floor_data['func_y'](x_local))
                slope = float(floor_data['func_slope'](x_local))
            except:
                state = "FALLING" # Error fallback
                continue

            # Slope Angle (local tangent angle)
            alpha_local = np.arctan(slope)
            
            # Total angle relative to gravity (Global Vertical)
            # Gravity acts down (-90 deg global). Slope is alpha + theta.
            # Component of gravity pulling along slope:
            # g_tangent = -g * sin(alpha_local + theta)
            g_force = -G * np.sin(alpha_local + theta)
            
            # Friction (Viscous damping for stability)
            f_force = -FRICTION * v_local
            
            # Update Local Physics
            accel = g_force + f_force
            v_local += accel * DT
            x_local += v_local * DT
            
            # Update Y based on track constraint
            y_local = float(floor_data['func_y'](x_local))
            
            # Calculate Global Position for Rendering
            x_glob, y_glob = to_global(x_local, y_local, theta)
            
            # --- CHECK BOUNDS (Did we fall off?) ---
            if x_local < floor_data['x_min'] or x_local > floor_data['x_max']:
                state = "FALLING"
                
                # TRANSFORM VELOCITY TO GLOBAL
                # 1. Relative Velocity Vector (tangent to local curve)
                # v_x_rel = v_local * cos(alpha) -> approximately just v_local for small slopes projected on x
                # A better approx: dx/dt = v_local (roughly), dy/dt = slope * v_local
                vx_rel_local = v_local
                vy_rel_local = slope * v_local
                
                # Rotate Relative Velocity to Global Frame
                vx_rel_glob, vy_rel_glob = to_global(vx_rel_local, vy_rel_local, theta)
                
                # 2. Add Tangential Velocity from Rotation (v = omega x r)
                # r vector is (x_glob, y_glob)
                # cross product in 2D: (-omega*y, omega*x)
                vx_tan = -omega * y_glob
                vy_tan = omega * x_glob
                
                # Total Global Launch Velocity
                vx_glob = vx_rel_glob + vx_tan
                vy_glob = vy_rel_glob + vy_tan
                
        
        # --- STATE: FALLING ---
        elif state == "FALLING":
            # Simple Ballistic Trajectory in Global Frame
            # Gravity acts purely on Global Y
            vy_glob -= G * DT
            
            x_glob += vx_glob * DT
            y_glob += vy_glob * DT
            
            # Check Collision with NEXT Floor(s)
            # We look at all floors to be safe, or just the next logic one
            # Transform global pos back to local to check height constraint
            x_loc_check, y_loc_check = to_local(x_glob, y_glob, theta)
            
            # Check if we are inside the domain of the next floor (or current)
            # Usually we fall to floor + 1
            next_idx = current_floor_idx + 1
            if next_idx in data['floors']:
                next_floor = data['floors'][next_idx]
                
                # Are we within x-bounds?
                if next_floor['x_min'] <= x_loc_check <= next_floor['x_max']:
                    target_y = float(next_floor['func_y'](x_loc_check))
                    
                    # Did we cross the line? (Current Y < Track Y) and close enough
                    if y_loc_check <= target_y + 0.02 and y_loc_check >= target_y - 0.1:
                        # IMPACT / LANDING
                        state = "ROLLING"
                        current_floor_idx = next_idx
                        floor_data = next_floor
                        x_local = x_loc_check
                        y_local = target_y
                        
                        # Energy dampening on landing
                        # Convert global velocity back to local approx
                        # We just take magnitude and dampen it
                        v_mag = np.sqrt(vx_glob**2 + vy_glob**2)
                        # Direction determines sign of new v_local (if moving right globally?)
                        # Simplifying: dot product with new tangent
                        slope = float(floor_data['func_slope'](x_local))
                        # Project global v onto global tangent
                        glob_slope_angle = np.arctan(slope) + theta
                        
                        # Scalar projection
                        v_local = (vx_glob * np.cos(glob_slope_angle) + 
                                   vy_glob * np.sin(glob_slope_angle))
                        
                        v_local *= 0.8 # Restitution coef (bounce dampening)

            # Safety check: fell off world
            if y_glob < -0.5:
                break

        # Record History
        history['x'].append(x_glob)
        history['y'].append(y_glob)
        history['angle'].append(theta)
        history['state'].append(state)

    return history

# --- 5. ANIMATION ---

def animate(history, sim_data):
    if not history['x']:
        print("No simulation data generated.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Create line objects for all visual segments
    lines = []
    for _ in sim_data['visuals']:
        ln, = ax.plot([], [], 'k-', lw=1)
        lines.append(ln)
        
    ball, = ax.plot([], [], 'ro', ms=6, zorder=10)
    title = ax.set_title("")

    def update(frame):
        angle = history['angle'][frame]
        
        # Update all floor/visual lines based on current rotation
        for i, (lx, ly) in enumerate(sim_data['visuals']):
            gx, gy = to_global(lx, ly, angle)
            lines[i].set_data(gx, gy)
            
        # Update ball
        ball.set_data([history['x'][frame]], [history['y'][frame]])
        title.set_text(f"State: {history['state'][frame]}")
        
        return lines + [ball, title]

    # Skip frames for speed (interval in ms)
    ani = FuncAnimation(fig, update, frames=range(0, len(history['x']), 2), 
                        interval=10, blit=True)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # IMPORTANT: Ensure 'dificultad1.json' and 'circulos.json' are in the working directory
    # or create dummy files here for testing.
    
    processed_data = process_json_data('dificultad1.json', 'circulos.json')
    
    if processed_data:
        print("Data loaded successfully. Starting simulation...")
        hist = run_simulation(processed_data)
        print(f"Simulation finished with {len(hist['x'])} frames.")
        animate(hist, processed_data)