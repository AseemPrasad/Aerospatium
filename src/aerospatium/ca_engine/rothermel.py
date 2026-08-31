import math

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(fastmath=True)
def rothermel_forward_ros(
    fuel_model: int,
    slope_rad: float,
    aspect_rad: float,
    wind_speed_ms: float,
    wind_dir_rad: float,
    neighbor_azimuth_rad: float,
    fuel_moisture: float = 0.08
) -> float:
    """
    Directional Rate of Spread (ROS) in m/min from burning cell to neighbor cell.
    Implements standard Rothermel (1972) / Albini (1976) surface fire behavior model.
    """
    if fuel_model == 99 or fuel_model <= 0:
        return 0.0

    # 1. Anderson 13 Standard Parameters
    # w0: kg/m2, delta: m, sigma_ft: 1/ft, mx: fraction, waf: midflame factor
    if fuel_model == 1:       # Short Grass (FM1)
        w0 = 0.166
        delta = 0.3048
        sigma_ft = 3500.0
        mx = 0.12
        waf = 0.4
    elif fuel_model == 2:     # Timber Grass (FM2)
        w0 = 0.896
        delta = 0.3048
        sigma_ft = 3000.0
        mx = 0.15
        waf = 0.3
    elif fuel_model == 4:     # Chaparral (FM4)
        w0 = 1.120
        delta = 1.8288
        sigma_ft = 2000.0
        mx = 0.20
        waf = 0.4
    elif fuel_model == 10:    # Timber Litter (FM10)
        w0 = 1.346
        delta = 0.3048
        sigma_ft = 2000.0
        mx = 0.25
        waf = 0.2
    else:
        w0, delta, sigma_ft, mx, waf = 0.166, 0.3048, 3500.0, 0.12, 0.4

    if fuel_moisture >= mx:
        return 0.0

    # 2. Fuel Bed Properties
    h = 18600.0                # Heat content (kJ/kg)
    rho_p = 512.0              # Particle density (kg/m3)
    rho_b = w0 / delta         # Bulk density (kg/m3)
    beta = rho_b / rho_p       # Packing ratio
    beta_op = 0.20395 * (sigma_ft ** (-0.8189))

    # 3. Damping Coefficients
    r_m = fuel_moisture / mx
    eta_m = max(0.0, min(1.0, 1.0 - 2.59 * r_m + 5.11 * (r_m ** 2) - 3.52 * (r_m ** 3)))
    eta_s = 0.174 * (0.0555 ** (-0.19))

    # 4. Reaction Intensity (I_R in kJ / m2-min)
    gamma_max = (sigma_ft ** 1.5) / (495.0 + 0.0594 * (sigma_ft ** 1.5))
    a = 8.9033 * (sigma_ft ** (-0.7913))
    gamma = gamma_max * ((beta / beta_op) ** a) * math.exp(a * (1.0 - (beta / beta_op)))
    i_r = gamma * w0 * h * eta_m * eta_s

    # 5. Propagating Flux Ratio (xi)
    xi = math.exp((0.792 + 0.681 * (sigma_ft ** 0.5)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma_ft)

    # 6. Heat of Pre-ignition (Q_ig) & Effective Heating Number (epsilon)
    epsilon = math.exp(-138.0 / sigma_ft)
    q_ig = 581.0 + 2594.0 * fuel_moisture  # kJ/kg

    # Base Rate of Spread (m/min)
    r0 = (i_r * xi) / (rho_b * epsilon * q_ig)

    # 7. Directional Wind Factor (Phi_w) with Rothermel / Albini Upper Bound
    # Convert midflame wind to ft/min
    midflame_u_fpm = (wind_speed_ms * waf) * 196.85
    wind_alignment = math.cos(wind_dir_rad - neighbor_azimuth_rad)
    effective_u_fpm = max(0.0, midflame_u_fpm * wind_alignment)

    # Upper bound constraint on effective wind speed (Albini 1976: U_max = 0.9 * I_R in BTU/ft2-min or ~25 m/s)
    max_u_fpm = min(4400.0, 0.9 * (i_r * 0.088055))
    effective_u_fpm = min(effective_u_fpm, max_u_fpm)

    if effective_u_fpm > 0.0:
        c_coef = 7.47 * (sigma_ft ** (-0.55))
        b_coef = 0.02526 * (sigma_ft ** 0.54)
        e_coef = 0.715 * math.exp(-0.000359 * sigma_ft)
        phi_w = c_coef * (effective_u_fpm ** b_coef) * ((beta / beta_op) ** (-e_coef))
        # Cap excessive wind amplification factor to max physical wildfire limit (~30x for severe wind)
        phi_w = min(phi_w, 35.0)
    else:
        phi_w = 0.0

    # 8. Directional Slope Factor (Phi_s)
    slope_alignment = math.cos(aspect_rad - neighbor_azimuth_rad)
    effective_tan_slope = max(0.0, math.tan(slope_rad) * slope_alignment)
    phi_s = 5.275 * (beta ** (-0.3)) * (effective_tan_slope ** 2)
    phi_s = min(phi_s, 25.0)

    # Directional forward spread rate (m/min)
    ros_m_min = r0 * (1.0 + phi_w + phi_s)
    return max(0.0, ros_m_min)
