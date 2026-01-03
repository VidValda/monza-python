import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

class Level:
    def __init__(self, diff_path, circ_path):
        self.floors = {}
        self.visuals = []
        self._load_data(diff_path, circ_path)

    def _load_data(self, d_path, c_path):
        try:
            with open(d_path, 'r') as f: d_data = json.load(f)
            with open(c_path, 'r') as f: c_data = json.load(f)
        except FileNotFoundError:
            return

        for i in range(0, 20):
            xk, yk = f'xp{i}', f'yp{i}'
            if i == 0:
                xk, yk = 'xp','yp'
                       
            if xk in d_data and yk in d_data:
                x = np.array(d_data[xk]).flatten()
                y = np.array(d_data[yk]).flatten()
                idx = np.argsort(x)
                x, y = x[idx], y[idx]
                
                dx = np.gradient(x)
                dx[dx == 0] = 1e-9
                slope = np.gradient(y) / dx
                
                self.floors[i] = {
                    'func': interp1d(x, y, kind='cubic', fill_value="extrapolate"),
                    'slope': interp1d(x, slope, kind='linear', fill_value="extrapolate"),
                    'min': x[0], 
                    'max': x[-1]
                }
                self.visuals.append((x, y))

        i = 1
        while True:
            xk, yk = f'xl{i}', f'yl{i}'
            if xk not in d_data: break
            self.visuals.append((np.array(d_data[xk]).flatten(), np.array(d_data[yk]).flatten()))
            i += 1

        r_keys = sorted([k for k in c_data if k.startswith('r')], key=lambda x: int(x[1:]))
        for i in range(0, len(r_keys), 2):
            if i + 1 < len(r_keys):
                self.visuals.append((np.array(c_data[r_keys[i]]).flatten(), np.array(c_data[r_keys[i+1]]).flatten()))

class PhysicsBall:
    def __init__(self, level):
        self.lvl = level
        self.idx = 2
        self.state = "ROLLING"
        self.lx = 0.001        
        self.ly = float(self.lvl.floors[self.idx]['func'](self.lx))


        self.lv = 0.0
        self.gx, self.gy = 0.0, 0.0
        self.gvx, self.gvy = 0.0, 0.0
        self.params = {'dt': 0.01, 'g': 9.81, 'fric': 0.05, 'rest': 0.6}

    def _to_global(self, lx, ly, ang):
        c, s = np.cos(ang), np.sin(ang)
        return lx * c - ly * s, lx * s + ly * c

    def _to_local(self, gx, gy, ang):
        c, s = np.cos(ang), np.sin(ang)
        return gx * c + gy * s, -gx * s + gy * c

    def update(self, angle, d_angle_dt):
        sub_steps = 10
        dt = self.params['dt'] / sub_steps
        g = self.params['g']

        for _ in range(sub_steps):
            if self.state == "ROLLING":
                floor = self.lvl.floors[self.idx]
                slope = float(floor['slope'](self.lx))
                alpha = np.arctan(slope)
                
                accel = -g * np.sin(alpha + angle) - self.params['fric'] * self.lv
                self.lv += accel * dt
                self.lx += self.lv * dt
                self.ly = float(floor['func'](self.lx))
                
                self.gx, self.gy = self._to_global(self.lx, self.ly, angle)
                
                if self.lx < floor['min'] or self.lx > floor['max']:
                    self.state = "FALLING"
                    vx_loc = self.lv
                    vy_loc = slope * self.lv
                    
                    gvx_rel, gvy_rel = self._to_global(vx_loc, vy_loc, angle)
                    vx_tan = -d_angle_dt * self.gy
                    vy_tan = d_angle_dt * self.gx
                    
                    self.gvx, self.gvy = gvx_rel + vx_tan, gvy_rel + vy_tan

            elif self.state == "FALLING":
                self.gvy -= g * dt
                self.gx += self.gvx * dt
                self.gy += self.gvy * dt
                
                for f_idx, floor in self.lvl.floors.items():
                    if f_idx == self.idx and self.state == "ROLLING": continue 

                    lx_chk, ly_chk = self._to_local(self.gx, self.gy, angle)
                    
                    if floor['min'] <= lx_chk <= floor['max']:
                        target_y = float(floor['func'](lx_chk))
                        
                        if ly_chk <= target_y:
                            slope = float(floor['slope'](lx_chk))
                            floor_angle = np.arctan(slope) + angle
                            
                            nx = -np.sin(floor_angle)
                            ny = np.cos(floor_angle)
                            
                            v_dot_n = self.gvx * nx + self.gvy * ny
                            
                            if v_dot_n < 0:
                                j = -(1 + self.params['rest']) * v_dot_n
                                self.gvx += j * nx
                                self.gvy += j * ny
                                
                                self.gx += nx * (target_y - ly_chk + 0.001)
                                self.gy += ny * (target_y - ly_chk + 0.001)

                                if abs(v_dot_n) < 0.5:
                                    self.state = "ROLLING"
                                    self.idx = f_idx
                                    self.lx = lx_chk
                                    self.ly = target_y
                                    
                                    tx = np.cos(floor_angle)
                                    ty = np.sin(floor_angle)
                                    self.lv = self.gvx * tx + self.gvy * ty
                                    break

        return self.gx, self.gy

def run_visuals(level, ball):
    hist = {'x': [], 'y': [], 'angle': []}
    steps = 600
    
    for s in range(steps):
        t = s * 0.01
        angle = 0.2 * np.sin(4 * t)
        omega = 0.2 * np.cos(4 * t)
        
        gx, gy = ball.update(angle, omega)
        
        hist['x'].append(gx)
        hist['y'].append(gy)
        hist['angle'].append(angle)
        
        if gy < -1.0: break

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    lines = [ax.plot([], [], 'k-', lw=1)[0] for _ in level.visuals]
    dot, = ax.plot([], [], 'ro', markersize=6, zorder=10)
    trace, = ax.plot([], [], 'r-', lw=0.5, alpha=0.5)

    def anim(f):
        a = hist['angle'][f]
        for ln, (lx, ly) in zip(lines, level.visuals):
            gx, gy = ball._to_global(lx, ly, a)
            ln.set_data(gx, gy)
        
        dot.set_data([hist['x'][f]], [hist['y'][f]])        
        trace.set_data(hist['x'][:f], hist['y'][:f])
        return lines + [dot, trace]

    ani = FuncAnimation(fig, anim, frames=len(hist['x']), interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    lvl = Level('dificultad1.json', 'circulos.json')
    if lvl.floors:
        sim = PhysicsBall(lvl)
        run_visuals(lvl, sim)