from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#States for the FSM
RIGHT, UP, LEFT, DOWN, DONE = 0, 1, 2, 3, 4
MOVE_TO_PARK, WIPE, COOLDOWN, LOAD, HEAT, RETURN = 5, 6, 7, 8, 9, 10

#Paremeters with sensible assigned values
@dataclass
class Params:
    m: float = 1.0     # mass
    rm: float = 0.05   # Motor pulley radius
    Km: float = 0.1    # Motor constant
    Jm: float = 0.01   # Motor inertia
    Lm: float = 0.05   # Motor inductance
    Rm: float = 2.0    # Motor resistance

    Jtot: float = Jm + 0.5 * m * rm**2  # Total inertia

    Cn: float = 1e-3
    Rn: float = 10.0
    Ln: float = 1e-4
    Ct_n: float = 5.0
    Rt_n: float = 20.0
    Te: float = 20.0   # Environment temperature

    v_ref: float = 0.5  # Reference velocity

    saved_state: int = RIGHT
    interrupt_pos_x: float = 0.0
    interrupt_pos_y: float = 0.0


def f(t, x, q, p):
    pos_x, vel_x, i_x, pos_y, vel_y, i_y, T_n, v_c, i_L = x

    u_xy = (p.Km / p.rm) * p.v_ref
    u_n = 10

    if q == RIGHT:
        u_x = u_xy
        u_y = 0.0
    elif q == UP:
        u_x = 0.0
        u_y = u_xy
    elif q == LEFT:
        u_x = -u_xy
        u_y = 0.0
    elif q == DOWN:
        u_x = 0.0
        u_y = -u_xy


    elif q == MOVE_TO_PARK:
        dist = np.sqrt(pos_x**2 + pos_y**2)
        if dist > 1e-4:
            u_x = -u_xy * (pos_x / dist)
            u_y = -u_xy * (pos_y / dist)
        else:
            u_x, u_y = 0.0, 0.0
    elif q == WIPE:
        u_x = 0.0
        u_y = -1.0
    elif q == COOLDOWN or q == LOAD:
        u_x = 0.0
        u_y = 0.0
        u_n = 0.0
    elif q == HEAT:
        u_x = 0.0
        u_y = 0.0
        u_n = 10.0
    elif q == RETURN:
        dx_target = p.interrupt_pos_x - pos_x
        dy_target = p.interrupt_pos_y - pos_y
        dist = np.sqrt(dx_target**2 + dy_target**2)
        if dist > 1e-4:
            u_x = u_xy * (dx_target / dist)
            u_y = u_xy * (dy_target / dist)
        else:
            u_x, u_y = 0.0, 0.0
    else:
        u_x = 0.0
        u_y = 0.0

    #State space models
    dx = vel_x
    d_vel_x = (p.rm * p.Km / p.Jtot) * i_x
    d_ix = -(p.Km / (p.rm * p.Lm)) * vel_x - (p.Rm / p.Lm) * i_x + (1 / p.Lm) * u_x

    dy = vel_y
    d_vel_y = (p.rm * p.Km / p.Jtot) * i_y
    d_iy = -(p.Km / (p.rm * p.Lm)) * vel_y - (p.Rm / p.Lm) * i_y + (1 / p.Lm) * u_y

    dT_n = (1 / (p.Ct_n * p.Rn)) * (v_c**2) - (1 / (p.Ct_n * p.Rt_n)) * (T_n - p.Te)
    d_vc = (1 / p.Cn) * i_L - (1 / (p.Cn * p.Rn)) * v_c
    d_iL = (1 / p.Ln) * u_n - (1 / p.Ln) * v_c

    return np.array([dx, d_vel_x, d_ix, dy, d_vel_y, d_iy, dT_n, d_vc, d_iL])


def make_events(q, p):
    events = []
    if q == RIGHT:
        def g_right(t, x):
            return x[0] - 1
        g_right.direction, g_right.terminal = 1, True
        events.append(g_right)
    elif q == UP:
        def g_up(t, x):
            return x[3] - 1
        g_up.direction, g_up.terminal = 1, True
        events.append(g_up)
    elif q == LEFT:
        def g_left(t, x):
            return x[0] - 0 
        g_left.direction, g_left.terminal = -1, True
        events.append(g_left)
    elif q == DOWN:
        def g_down(t, x):
            return x[3] - 0
        g_down.direction, g_down.terminal = -1, True
        events.append(g_down)


    elif q == MOVE_TO_PARK:
        def g_park(t, x):
            return x[0]**2 + x[3]**2 - 0.0001
        g_park.direction, g_park.terminal = -1, True
        events.append(g_park)
    elif q == COOLDOWN:
        def g_cooldown(t, x):
            return x[6] - 25
        g_cooldown.direction, g_cooldown.terminal = -1, True
        events.append(g_cooldown)
    elif q == LOAD:
        def g_load(t, x):
            return x[6] - 24
        g_load.direction, g_load.terminal = -1, True
        events.append(g_load)
    elif q == HEAT:
        def g_heat(t, x):
            return x[6] - 30
        g_heat.direction, g_heat.terminal = 1, True
        events.append(g_heat) 
    elif q == RETURN:
        def g_return(t, x): 
            return (x[0] - p.interrupt_pos_x)**2 + (x[3] - p.interrupt_pos_y)**2 - 0.0001
        g_return.direction, g_return.terminal = -1, True
        events.append(g_return)
    elif q == WIPE:
        def g_wipe(t, x):
            return x[3] + 0.1
        g_wipe.direction, g_wipe.terminal = -1, True
        events.append(g_wipe)
    return events




def jump(t, x_minus, q_minus, p):
    x_plus = np.copy(x_minus)

    x_plus[1] = 0.0  # Reset x velocity to zero
    x_plus[2] = 0.0  # Reset x current to zero
    x_plus[4] = 0.0  # Reset y velocity to zero
    x_plus[5] = 0.0  # Reset y current to zero
    x_plus[8] = 0.0  # Reset i_L to zero

    if q_minus == RIGHT:
        q_plus = UP
    elif q_minus == UP:
        q_plus = LEFT
    elif q_minus == LEFT:
        q_plus = DOWN
    elif q_minus == DOWN:
        q_plus = DONE
    elif q_minus == MOVE_TO_PARK:
        q_plus = WIPE
    elif q_minus == WIPE:
        q_plus = COOLDOWN
    elif q_minus == COOLDOWN:
        q_plus = LOAD
    elif q_minus == LOAD:
        q_plus = HEAT
    elif q_minus == HEAT:
        dist = (x_plus[0] - p.interrupt_pos_x)**2 + (x_plus[3] - p.interrupt_pos_y)**2
        if dist < 1e-4:
            q_plus = p.saved_state
        else:
            q_plus = RETURN
    elif q_minus == RETURN:
        q_plus = p.saved_state
        x_plus[0] = p.interrupt_pos_x
        x_plus[3] = p.interrupt_pos_y
    else:
        q_plus = DONE

    return x_plus, q_plus


def simulate(x0, p):
    x = np.asarray(x0, float)
    t = 0.0

    q = HEAT

    p.saved_state = RIGHT
    p.interrupt_pos_x = 0.0
    p.interrupt_pos_y = 0.0

    segs, log = [], []

    triggered = False
    filament_change_time = 15
    sim_time = 350

    while t < sim_time and q != DONE:
        t_span_end = filament_change_time if (not triggered and t < filament_change_time) else sim_time

        sol = solve_ivp(lambda tt, xx: f(tt, xx, q, p), (t, t_span_end), x,
                        events=make_events(q, p),
                        rtol=1e-9, atol=1e-12, max_step=0.05)

        segs.append((sol.t, sol.y, q))

        if sol.status == 1:
            t_event = sol.t_events[0][0]
            x_event = sol.y_events[0][0]

            x, q_new = jump(t_event, x_event, q, p)
            log.append((t_event, x, q, q_new))
            q = q_new
            t = t_event
        else:
            if not triggered and sol.t[-1] >= filament_change_time:
                triggered = True
                p.saved_state = q
                p.interrupt_pos_x = sol.y[0, -1]
                p.interrupt_pos_y = sol.y[3, -1]
                q = MOVE_TO_PARK
                x = sol.y[:, -1]
                x[1], x[2], x[4], x[5], x[8] = 0.0, 0.0, 0.0, 0.0, 0.0 
                t = sol.t[-1]
            else:
                break
    return segs, log



if __name__ == "__main__":
    p = Params()
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p.Te, 0.0, 0.0]

    segs, log = simulate(x0, p)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for t, y, q in segs:
        color = 'red' if q >= 5 else 'blue'
        ax1.plot(y[0], y[3], color=color)
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_title("Position Trajectory with FSM States")
    ax1.grid(True)

    for t, y, q in segs:
        color = 'red' if q >= 5 else 'blue'
        ax2.plot(t, y[6], color=color)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Temperature")
    ax2.set_title("Temperature vs Time with FSM States")
    ax2.grid(True)

    fig1.tight_layout()
    fig1.savefig("trajectory_and_temperature_with_states.png")


    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    for t, y, q in segs:
        color = 'red' if q >= 5 else 'blue'
        ax3.plot(t, y[1], color=color)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("X Velocity")
    ax3.set_title("X Velocity vs Time with FSM States")
    ax3.grid(True)

    for t, y, q in segs:
        color = 'red' if q >= 5 else 'blue'
        ax4.plot(t, y[2], color=color)
        ax4.plot(t, y[5], color=color)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Motor Currents")
    ax4.set_title("Motor Currents vs Time with FSM States")
    ax4.grid(True)

    fig2.tight_layout()
    fig2.savefig("velocity_and_currents_with_states.png")


    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    for t, y, q in segs:
        color = 'red' if q >= 5 else 'blue'
        q_array = np.full(len(t), q)
        ax5.plot(t, q_array, color=color)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("FSM State")
    ax5.set_title("FSM State vs Time")
    ax5.set_yticks(range(11))
    ax5.set_yticklabels(['RIGHT', 'UP', 'LEFT', 'DOWN', 'DONE', 'MOVE_TO_PARK', 'WIPE', 'COOLDOWN', 'LOAD', 'HEAT', 'RETURN'])
    ax5.grid(True)


    for t, y, q in segs:
        color = 'red' if q >= 5 else 'blue'
        ax6.plot(t, y[7], color=color)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Capacitor Voltage")
    ax6.set_title("Capacitor Voltage vs Time with FSM States")
    ax6.grid(True)

    fig3.tight_layout()
    fig3.savefig("fsm_state_and_capacitor_voltage.png")
