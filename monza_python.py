import numpy as np
import scipy.io as sio
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURATION & CONSTANTS ---
DT = 0.005
T_TOTAL = 60
STEPS = int(T_TOTAL / DT)
G = 9.8
M = 0.007
C_V = 0.01

# Offsets for each floor (Height difference constants from solver.m)
# Floor 1 falls to 2 (+0.0229 or similar offset logic)
# Based on solver.m logic:
OFFSETS = {
    1: 0.0686,   # Target floor 1? (Not used usually)
    2: 0.0229,   # Target floor 2
    3: -0.0229,  # Target floor 3
    4: -0.0686,  # Target floor 4
    5: -0.1143,  # Target floor 5
    6: 0.1600    # Target floor 6 (This looks like a distinct jump)
}
# Note: The solver.m snippet had specific offsets for specific cases. 
# We will match the switch-case logic in the solver function below.

def load_data():
    """Loads .mat files and organizes track geometry."""
    try:
        mat_diff = sio.loadmat('dificultad1.mat')
        mat_circ = sio.loadmat('circulos.mat')
        
        # Helper to extract and flatten arrays
        def get_arr(name):
            return mat_diff[name].flatten()
        
        # Organize track limits into a dictionary for easy access
        tracks = {}
        tracks[1] = get_arr('xp1')
        tracks[2] = get_arr('xp2')
        tracks[3] = get_arr('xp3')
        tracks[4] = get_arr('xp4')
        tracks[5] = get_arr('xp5')
        tracks[6] = get_arr('xp6')
        tracks[7] = get_arr('xp7')
        
        # Visual data
        visuals = {
            'xl': [get_arr(f'xl{i}') for i in range(1, 5)],
            'yl': [get_arr(f'yl{i}') for i in range(1, 5)],
            'xp': [get_arr(f'xp{i}') for i in range(1, 8)], # Specific tracks
            'yp': [get_arr(f'yp{i}') for i in range(1, 8)],
            'xp_base': get_arr('xp'), # Base track
            'yp_base': get_arr('yp'),
            'r': [mat_circ[f'r{i}'].flatten() for i in range(1, 5)]
        }
        return tracks, visuals
    except FileNotFoundError:
        print("Error: .mat files not found. Ensure 'dificultad1.mat' and 'circulos.mat' are present.")
        return None, None

def solver(a, b, v, ang_pos, x_resp, y_resp, target_floor):
    """
    Replicates the logic of solver.m to find the impact x-coordinate.
    Solves for xc where the ballistic trajectory intersects the parabola.
    """
    d = x_resp
    h = y_resp
    
    # Determine offset constant C based on solver.m
    # The solver.m uses: -0.54 * (...) + C
    C = 0.0
    if target_floor == 2: C = 0.0229
    elif target_floor == 3: C = -0.0229
    elif target_floor == 4: C = -0.0686
    elif target_floor == 5: C = -0.1143
    elif target_floor == 6: C = 0.1600 # From user snippet
    else: C = 0.0 # Default/Fallback
    
    # Physics constants for the equation
    # We want to find xc.
    # We need to replicate the equation: LHS - RHS = 0
    # LHS (Rotated Y of projectile): -b*xc + a* (Trajectory_Y_local)
    # RHS (Track Height): -0.54 * (Rotated X)^2 + C
    
    # Trajectory Y relative to launch d (in non-rotated frame relative to launch):
    # y_traj = (xc - d)*tan(ang) - (g*(xc-d)^2)/(2*v^2*cos(ang)^2) + h
    # BUT, 'xc' in solver.m seems to be the global X?
    # Let's look closely at solver.m provided:
    # solve( (-b*xc + a*(...)) == -0.54 * (a*xc + b*(...)).^2 + C )
    # It appears 'xc' is indeed the Global X coordinate of the collision.
    
    term_g = 9.8 / (2 * v**2 * np.cos(ang_pos)**2)
    tan_a = np.tan(ang_pos)
    
    def equations(xc):
        # The projectile height at global x = xc
        dx = xc - d
        y_proj = dx * tan_a - term_g * dx**2 + h
        
        # Convert Global (xc, y_proj) to Local Track Coordinates
        # x_local = xc * a + y_proj * b
        # y_local = -xc * b + y_proj * a
        # Wait, the solver.m inverse might be different depending on definition.
        # Standard rotation:
        # x_loc = x*cos + y*sin
        # y_loc = -x*sin + y*cos
        # In solver.m:
        # LHS (y_loc) = -b*xc + a*y_proj  (Matches -x*sin + y*cos)
        # RHS (Parabola) = -0.54 * (a*xc + b*y_proj)^2 + C  (Matches -0.54 * x_loc^2 + C)
        
        x_loc = xc * a + y_proj * b
        y_loc = -xc * b + y_proj * a
        
        return y_loc - (-0.54 * x_loc**2 + C)

    # Initial guess: slightly away from launch x
    xc_guess = d + 0.1
    if target_floor % 2 == 0: # Even floors usually to the left
        xc_guess = d - 0.1
        
    xc_solution = fsolve(equations, xc_guess)[0]
    
    # Calculate corresponding y and local coordinates
    dx = xc_solution - d
    y_solution = dx * tan_a - term_g * dx**2 + h
    
    x_local_hit = xc_solution * a + y_solution * b
    y_local_hit = -xc_solution * b + y_solution * a
    
    return xc_solution, y_solution, x_local_hit

def run_simulation():
    track_limits, visuals = load_data()
    if not track_limits: return

    # --- State Variables ---
    piso = 1
    # Start at the minimum X of floor 1 + epsilon
    x = np.min(track_limits[1]) + 0.001 
    dxt = 0.0 # Velocity
    
    cont = 100 # 100=Waiting, 0=Rolling, 1=Falling
    
    # Fall variables
    x_resp, y_resp = 0.0, 0.0
    vx_glob, vy_glob = 0.0, 0.0
    t_drop = 0.0
    target_x_glob, target_y_glob = 0.0, 0.0
    
    # Storage for animation
    history = {
        'x': [], 'y': [], 'piso': [], 'angle': []
    }
    
    print("Starting simulation...")
    
    for step in range(STEPS):
        t = step * DT
        
        # A. INPUT (Rotation)
        giro = 0.5 * np.sin(t * 0.5)
        a = np.cos(giro)
        b = np.sin(giro)
        
        # B. COORD TRANSFORMS
        y_local = -0.54 * x**2
        xo = x * a - y_local * b
        yo = x * b + y_local * a
        
        # C. STATE MACHINE
        if cont == 100:
            # WAITING STATE
            if step > 50: # Simple start delay
                cont = 0
                dxt = 0.05
            
            history['x'].append(xo)
            history['y'].append(yo)
            history['piso'].append(piso)
            history['angle'].append(giro)

        elif cont == 0:
            # ROLLING STATE
            theta = np.arctan(-1.08 * x)
            # Equations of motion
            # gravity term: -sin(theta + alpha)
            # friction: -cv * v
            d2xt = G * -np.sin(theta + giro) - (C_V / M) * dxt
            dxt += d2xt * DT
            x += dxt * DT
            
            # Limit Check (The "Deep Fix")
            # We use the loaded arrays exactly
            limits = track_limits[piso]
            min_x, max_x = np.min(limits), np.max(limits)
            
            falling = False
            
            # Tolerance 1mm
            if x > (max_x + 0.001) or x < (min_x - 0.001):
                falling = True
            
            # Safety for start/end
            if piso == 1 and x < min_x: 
                # Reset if trying to fall off start
                x = min_x + 0.001
                dxt = 0
                falling = False
            
            if piso == 7 and falling:
                print("Finished Track!")
                break

            if falling:
                cont = 1
                t_drop = 0
                
                # Calculate Launch Vectors
                vx_loc = dxt
                vy_loc = -1.08 * x * dxt
                
                vx_glob = vx_loc * a - vy_loc * b
                vy_glob = vx_loc * b + vy_loc * a
                
                x_resp, y_resp = xo, yo
                
                # CALL SOLVER
                v_mag = np.sqrt(vx_glob**2 + vy_glob**2)
                ang_v = np.arctan2(vy_glob, vx_glob)
                
                # Predict impact on next floor
                if piso < 7:
                    tx, ty, t_loc = solver(a, b, v_mag, ang_v, x_resp, y_resp, piso + 1)
                    target_x_glob, target_y_glob = tx, ty
                    # Store the local X impact to snap to it later
                    target_x_local_snap = t_loc
                else:
                    target_x_glob = 0 # Fall into abyss
            
            history['x'].append(xo)
            history['y'].append(yo)
            history['piso'].append(piso)
            history['angle'].append(giro)

        elif cont == 1:
            # FALLING STATE
            t_drop += DT
            
            xo = x_resp + vx_glob * t_drop
            yo = y_resp + vy_glob * t_drop - 0.5 * G * t_drop**2
            
            # Check Impact
            dist = np.sqrt((xo - target_x_glob)**2 + (yo - target_y_glob)**2)
            
            # Impact Logic: Distance < 5cm OR passed height
            if dist < 0.05 or (yo < target_y_glob - 0.02):
                if piso < 7:
                    cont = 0
                    piso += 1
                    
                    # Snap to track
                    x = target_x_local_snap
                    
                    # Energy Loss
                    v_impact = np.sqrt(vx_glob**2 + (vy_glob - G*t_drop)**2)
                    dxt = v_impact * 0.7
                    
                    # Direction logic (Odd floors right, Even left)
                    if piso in [2, 4, 6]:
                        dxt = -abs(dxt)
                    else:
                        dxt = abs(dxt)
                else:
                    # End of simulation
                    print("Fell off world.")
                    break
            
            history['x'].append(xo)
            history['y'].append(yo)
            history['piso'].append(piso)
            history['angle'].append(giro)

    return history, visuals

# --- 2. ANIMATION ---
def animate(history, visuals):
    if not history: return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Static Background (Gray Circles)
    for r in visuals['r']:
        x_vals = r[0::2]
        y_vals = r[1::2]
        # Ensure both arrays have the same length (handle odd-length arrays)
        min_len = min(len(x_vals), len(y_vals))
        ax.plot(x_vals[:min_len], y_vals[:min_len], color='#d3d3d3', linewidth=1, zorder=0) 
        # Note: r1 in matlab is often x,y pairs or radius. Assuming plot(x,y).
        # Adjust indexing based on actual mat structure if needed.
        # Often circulos.mat has r1 as X and r2 as Y.
    
    # Plot objects that will update
    line_objects = []
    
    # We have multiple segments to draw:
    # 4 straight lines (xl) and 7+1 parabolas (xp)
    # We create a list of Line2D objects
    num_segments = len(visuals['xl']) + len(visuals['xp']) + 1 # +1 for base
    for _ in range(num_segments):
        ln, = ax.plot([], [], 'b-', linewidth=1.5)
        line_objects.append(ln)
        
    coin_dot, = ax.plot([], [], 'ro', markersize=8, zorder=10)
    title_text = ax.set_title("")

    def rotate(x, y, ang):
        ca, cb = np.cos(ang), np.sin(ang)
        rx = x * ca - y * cb
        ry = x * cb + y * ca
        return rx, ry

    def update(frame):
        # Current angle
        ang = history['angle'][frame]
        
        idx = 0
        
        # Draw Straight Segments (xl1..xl4)
        for i in range(len(visuals['xl'])):
            rx, ry = rotate(visuals['xl'][i], visuals['yl'][i], ang)
            line_objects[idx].set_data(rx, ry)
            idx += 1
            
        # Draw Base Track (xp)
        rx, ry = rotate(visuals['xp_base'], visuals['yp_base'], ang)
        line_objects[idx].set_data(rx, ry)
        idx += 1
        
        # Draw Floor Tracks (xp1..xp7)
        for i in range(len(visuals['xp'])):
            rx, ry = rotate(visuals['xp'][i], visuals['yp'][i], ang)
            line_objects[idx].set_data(rx, ry)
            idx += 1
            
        # Draw Coin
        coin_dot.set_data([history['x'][frame]], [history['y'][frame]])
        
        title_text.set_text(f"Floor: {history['piso'][frame]} | Time: {frame*DT:.2f}s")
        return line_objects + [coin_dot, title_text]

    # Create Animation
    # Skip frames for speed (interval=20ms roughly 50fps)
    ani = FuncAnimation(fig, update, frames=range(0, len(history['x']), 5),
                        interval=20, blit=True)
    
    plt.show()

if __name__ == "__main__":
    hist, vis = run_simulation()
    if hist:
        animate(hist, vis)