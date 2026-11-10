"""
2D Aerodynamic Panel Flow Solver
A computational fluid dynamics (CFD) potential flow solver using the source panel method.
Computes velocity fields, streamlines, and pressure coefficient (Cp) distributions 
around arbitrary 2D aerodynamic and blunt geometries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import integrate

# --- CONFIGURATION & PARAMETERS ---
U_inf = 10.0           # Freestream velocity [m/s]
alpha = 0.0            # Angle of attack [degrees] (Default: 0 for direct headwind)
N_panels = 40          # Number of boundary panels for discretization

# Grid Domain setup
x_start, x_end = -4.0, 4.0
y_start, y_end = -2.0, 2.0
N_grid = 100
X, Y = np.meshgrid(np.linspace(x_start, x_end, N_grid), np.linspace(y_start, y_end, N_grid))

# --- PANEL DISCRETIZATION CLASS ---
class Panel:
    """Defines a straight line segment acting as a source sheet."""
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        self.xc, self.yc = (xa + xb) / 2.0, (ya + yb) / 2.0  # Control point
        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)
        self.phi = np.arctan2((yb - ya), (xb - xa))          # Orientation angle
        self.beta = self.phi + np.pi / 2                   # Normal vector angle
        self.sigma = 0.0                                   # Source strength

def define_panels(x, y, N):
    """Discretizes an arbitrary geometry into N uniform panels."""
    if not (x[0] == x[-1] and y[0] == y[-1]):
        x, y = np.append(x, x[0]), np.append(y, y[0])  # Close boundary contour
        
    distance = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative = np.insert(np.cumsum(distance), 0, 0.0)
    
    fx, fy = interp1d(cumulative, x), interp1d(cumulative, y)
    starts = np.linspace(0.0, cumulative[-1], N + 1)
    
    return [Panel(fx(starts[i]), fy(starts[i]), fx(starts[i + 1]), fy(starts[i + 1])) for i in range(N)]

def solve_source_strengths(panels):
    """Solves the linear system for panel source strengths (sigma)."""
    A = np.empty((N_panels, N_panels), dtype=float)
    b = -U_inf * np.array([np.cos(alpha - p.beta) for p in panels])
    
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i == j:
                A[i, j] = 0.5
            else:
                # Analytic integral of influence of panel j on control point of panel i
                xp = (p_i.xc - p_j.xa) * np.cos(p_j.phi) + (p_i.yc - p_j.ya) * np.sin(p_j.phi)
                yp = -(p_i.xc - p_j.xa) * np.sin(p_j.phi) + (p_i.yc - p_j.ya) * np.cos(p_j.phi)
                
                f = lambda s: yp / ((xp - s)**2 + yp**2)
                A[i, j] = (1 / (2 * np.pi)) * integrate.quad(f, 0.0, p_j.length)[0]
                
    sigma_solution = np.linalg.solve(A, b)
    for i, panel in enumerate(panels):
        panel.sigma = sigma_solution[i]
    return panels

def compute_velocity_field(panels):
    """Calculates the velocity vector field (u, v) over the defined mesh grid."""
    u = U_inf * np.cos(alpha) * np.ones_like(X)
    v = U_inf * np.sin(alpha) * np.ones_like(Y)
    
    for p in panels:
        dx, dy = X - p.xa, Y - p.ya
        xp = dx * np.cos(p.phi) + dy * np.sin(p.phi)
        yp = -dx * np.sin(p.phi) + dy * np.cos(p.phi)
        
        # Induced velocity components in local coordinates
        up_local = (p.sigma / (4 * np.pi)) * np.log((xp**2 + yp**2) / ((xp - p.length)**2 + yp**2))
        vp_local = (p.sigma / (2 * np.pi)) * (np.arctan2(yp, xp - p.length) - np.arctan2(yp, xp))
        
        # Transform velocity back to global Cartesian coordinates
        u += up_local * np.cos(p.phi) - vp_local * np.sin(p.phi)
        v += up_local * np.sin(p.phi) + vp_local * np.cos(p.phi)
        
    return u, v

def plot_streamlines(panels, u, v, filename):
    """Generates and saves streamline vector visualization."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
    ax.fill([p.xa for p in panels], [p.ya for p in panels], 'gray')
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    ax.set_aspect('equal')
    plt.title('Aerodynamic Streamlines')
    plt.savefig(f"assets/{filename}_streamlines.png")
    plt.close()

def plot_pressure_coefficient(panels, u, v, filename):
    """Computes and plots the dimensionless Pressure Coefficient (Cp) map."""
    cp = 1.0 - (u**2 + v**2) / U_inf**2
    
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = np.linspace(-2.0, 2.0, 100)
    contour = ax.contourf(X, Y, cp, levels=levels, extend='both', cmap='jet')
    fig.colorbar(contour, label=r'Pressure Coefficient $C_p$')
    
    ax.fill([p.xa for p in panels], [p.ya for p in panels], 'white')
    ax.plot([p.xa for p in panels], [p.ya for p in panels], 'k-', lw=2)
    ax.set_title('Pressure Coefficient Map')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.savefig(f"assets/{filename}_cp_map.png")
    plt.close()

def main():
    print("Executing 2D Panel Method Solver...")
    
    # Baseline test geometry: 2D aerodynamic ellipse profile
    theta = np.linspace(0, 2 * np.pi, N_panels)
    x_geom = np.cos(theta)
    y_geom = np.sin(theta) * 0.2
    
    panels = define_panels(x_geom, y_geom, N_panels)
    panels = solve_source_strengths(panels)
    u, v = compute_velocity_field(panels)
    
    plot_streamlines(panels, u, v, "aerodynamic_profile")
    plot_pressure_coefficient(panels, u, v, "aerodynamic_profile")
    print("Simulation complete. Visualizations saved to assets/ folder.")

if __name__ == "__main__":
    main()
