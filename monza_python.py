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
        except FileNotFoundError: return

        for i in range(1, 20):
            xk, yk = f'xp{i}', f'yp{i}'
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
                    'min': x[0], 'max': x[-1]
                }
                self.visuals.append((x, y))

        for i in range(1, 20):
            xk, yk = f'xl{i}', f'yl{i}'
            if xk in d_data and yk in d_data:
                self.visuals.append((np.array(d_data[xk]).flatten(), np.array(d_data[yk]).flatten()))

        if 'xp' in d_data and 'yp' in d_data:
            self.visuals.append((np.array(d_data['xp']).flatten(), np.array(d_data['yp']).flatten()))

        r_keys = sorted([k for k in c_data if k.startswith('r')], key=lambda x: int(x[1:]))
        for i in range(0, len(r_keys), 2):
            if i + 1 < len(r_keys):
                self.visuals.append((np.array(c_data[r_keys[i]]).flatten(), np.array(c_data[r_keys[i+1]]).flatten()))

class PhysicsBall:
    def __init__(self, level):
        self.lvl = level
        self.idx = 1
        self.state = "ROLLING"
        self.lx = 0.005
        self.ly = float(self.lvl.floors[1]['func'](self.lx)) if 1 in self.lvl.floors else 0
        self.lv = 0.0
        self.gx, self.gy = 0.0, 0.0
        self.gvx, self.gvy = 0.0, 0.0
        self.params = {'dt': 0.01, 'g': 9.81, 'fric': 0.05}

    def _to_global(self, lx, ly, ang):
        c, s = np.cos(ang), np.sin(ang)
        return lx * c - ly * s, lx * s + ly * c

    def _to_local(self, gx, gy, ang):
        c, s = np.cos(ang), np.sin(ang)
        return gx * c + gy * s, -gx * s + gy * c

    def update(self, angle, d_angle_dt):
        dt, g = self.params['dt'], self.params['g']
        
        if self.state == "ROLLING":
            floor = self.lvl.floors[self.idx]
            self.ly = float(floor['func'](self.lx))
            slope = float(floor['slope'](self.lx))
            alpha = np.arctan(slope)
            
            accel = -g * np.sin(alpha + angle) - self.params['fric'] * self.lv
            self.lv += accel * dt
            self.lx += self.lv * dt
            
            self.gx, self.gy = self._to_global(self.lx, self.ly, angle)
            
            if self.lx < floor['min'] or self.lx > floor['max']:
                self.state = "FALLING"
                vx_rel, vy_rel = self.lv, slope * self.lv
                gvx_rel, gvy_rel = self._to_global(vx_rel, vy_rel, angle)
                vx_tan, vy_tan = -d_angle_dt * self.gy, d_angle_dt * self.gx
                self.gvx, self.gvy = gvx_rel + vx_tan, gvy_rel + vy_tan

        elif self.state == "FALLING":
            self.gvy -= g * dt
            self.gx += self.gvx * dt
            self.gy += self.gvy * dt
            
            next_idx = self.idx + 1
            if next_idx in self.lvl.floors:
                floor = self.lvl.floors[next_idx]
                lx_chk, ly_chk = self._to_local(self.gx, self.gy, angle)
                
                if floor['min'] <= lx_chk <= floor['max']:
                    ty = float(floor['func'](lx_chk))
                    if ly_chk <= ty + 0.05 and ly_chk >= ty - 0.1:
                        self.state = "ROLLING"
                        self.idx = next_idx
                        self.lx, self.ly = lx_chk, ty
                        
                        slope = float(floor['slope'](self.lx))
                        glob_slope = np.arctan(slope) + angle
                        self.lv = (self.gvx * np.cos(glob_slope) + self.gvy * np.sin(glob_slope)) * 0.8

        return self.gx, self.gy

def run_visuals(level, ball):
    hist = {'x': [], 'y': [], 'angle': [], 'state': []}
    steps = 1000
    
    for s in range(steps):
        t = s * 0.01
        angle = 0.5 * np.sin(0.5 * t)
        omega = 0.5 * 0.5 * np.cos(0.5 * t)
        
        gx, gy = ball.update(angle, omega)
        
        hist['x'].append(gx)
        hist['y'].append(gy)
        hist['angle'].append(angle)
        hist['state'].append(ball.state)
        
        if gy < -0.5: break

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.3, 0.3); ax.set_aspect('equal')
    lines = [ax.plot([], [], 'k-', lw=1)[0] for _ in level.visuals]
    dot, = ax.plot([], [], 'ro', zorder=10)

    def anim(f):
        a = hist['angle'][f]
        for ln, (lx, ly) in zip(lines, level.visuals):
            gx, gy = ball._to_global(lx, ly, a)
            ln.set_data(gx, gy)
        dot.set_data([hist['x'][f]], [hist['y'][f]])
        return lines + [dot]

    FuncAnimation(fig, anim, frames=range(0, len(hist['x']), 2), interval=10, blit=True)
    plt.show()

if __name__ == "__main__":
    lvl = Level('dificultad1.json', 'circulos.json')
    if lvl.floors:
        sim = PhysicsBall(lvl)
        run_visuals(lvl, sim)