import math
import pandas as pd
import streamlit as st
# --- Constants ---
G = 9.81  # Gravitational acceleration (m/s^2)
BBL_DAY_TO_M3_S = 0.158987 / (24 * 3600)
INCHES_TO_M = 0.0254
MM_TO_M = 0.001
KM_TO_M = 1000
PA_TO_BAR = 1 / 100000
LTR_MIN_TO_M3_S = 0.001 / 60
DEFAULT_EMISSION_FACTOR_KG_CO2_KWH = 0.5 # Example value

# Validated against: https://www.copely.com/discover/tools/pressure-drop-calculator/

# --- Data Structures ---
class PipelineParams:
    def __init__(self, length_m, diameter_m, roughness_m, elevation_gain_m):
        self.length_m = length_m
        self.diameter_m = diameter_m
        self.roughness_m = roughness_m
        self.elevation_gain_m = elevation_gain_m

class FluidProperties:
    def __init__(self, density_kgm3, viscosity_pas):
        self.density_kgm3 = density_kgm3
        self.viscosity_pas = viscosity_pas

class OperatingConditions:
    def __init__(self, flow_rate_m3s, pump_efficiency_decimal):
        self.flow_rate_m3s = flow_rate_m3s
        self.pump_efficiency_decimal = pump_efficiency_decimal

class Results:
    def __init__(self, pressure_drop_pa, pump_power_kw, energy_kwh_day=None, ghg_kg_co2e_day=None, velocity_ms=None, reynolds_number=None, friction_factor=None, pressure_drop_bar=None):
        self.pressure_drop_pa = pressure_drop_pa
        self.pressure_drop_bar = pressure_drop_pa * PA_TO_BAR if pressure_drop_pa is not None else None
        self.pump_power_kw = pump_power_kw
        self.energy_kwh_day = energy_kwh_day
        self.ghg_kg_co2e_day = ghg_kg_co2e_day
        # Intermediate results for info
        self.velocity_ms = velocity_ms
        self.reynolds_number = reynolds_number
        self.friction_factor = friction_factor


# --- Core Logic ---

def calculate_velocity(flow_rate_m3s, diameter_m):
    """Calculates fluid velocity."""
    if diameter_m <= 0:
        return 0
    area = math.pi * (diameter_m / 2)**2
    return flow_rate_m3s / area

def calculate_reynolds_number(density_kgm3, velocity_ms, diameter_m, viscosity_pas):
    """Calculates the Reynolds number."""
    if viscosity_pas <= 0 or diameter_m <= 0:
        return 0
    return (density_kgm3 * velocity_ms * diameter_m) / viscosity_pas

def calculate_friction_factor(reynolds_number, roughness_m, diameter_m):
    """Calculates the Darcy friction factor."""
    if reynolds_number <= 0 or diameter_m <= 0: return 0

    if reynolds_number < 2300:  # Laminar flow
        return 64 / reynolds_number
    # elif reynolds_number < 4000: # Transitional flow - complex, use turbulent approx for simplicity here
    #     # Could implement a more sophisticated transition model if needed
    #     pass

    # Turbulent flow: Using Swamee-Jain approximation
    relative_roughness = roughness_m / diameter_m
    if relative_roughness <= 0: # Avoid log(0) or log(<0)
        # Use smooth pipe correlation implicitly within Swamee-Jain by setting a very small roughness
         relative_roughness = 1e-9 # Or handle smooth pipe separately

    # Prevent math domain errors for log10
    term1_log_arg = (relative_roughness / 3.7)
    term2_log_arg = (5.74 / (reynolds_number**0.9))

    # Ensure arguments to log10 are positive
    if term1_log_arg + term2_log_arg <= 0:
        # This case is unlikely with physical inputs but handle defensively
        # Fallback or specific handling might be needed depending on requirements
        # For now, return a default or signal an error. Let's use a small default friction factor.
        # Or consider using Colebrook-White iterative solution which might be more robust.
        st.warning(f"Warning: Cannot calculate friction factor with Swamee-Jain for Re={reynolds_number:.2f}, RelRoughness={relative_roughness:.2e}. Using fallback.")
        # Fallback might be needed, e.g., Haaland equation or simpler approximation
        return 0.02 # Arbitrary fallback for now

    log_term = math.log10(term1_log_arg + term2_log_arg)
    if log_term == 0: return 0 # Avoid division by zero
    friction_factor = 0.25 / (log_term**2)

    return friction_factor


def calculate_pressure_drop(pipeline: PipelineParams, fluid: FluidProperties, velocity_ms: float, friction_factor: float):
    """Calculates the total pressure drop (friction + elevation)."""
    if pipeline.diameter_m <= 0: return 0

    # Friction loss (Darcy-Weisbach)
    pressure_drop_friction_pa = friction_factor * (pipeline.length_m / pipeline.diameter_m) * (fluid.density_kgm3 * velocity_ms**2) / 2

    # Elevation head loss/gain
    pressure_drop_elevation_pa = fluid.density_kgm3 * G * pipeline.elevation_gain_m

    total_pressure_drop_pa = pressure_drop_friction_pa + pressure_drop_elevation_pa
    return total_pressure_drop_pa

def calculate_pump_power(flow_rate_m3s, pressure_drop_pa, pump_efficiency_decimal):
    """Calculates the required pump power."""
    if pump_efficiency_decimal <= 0:
        return 0 # Avoid division by zero, power is infinite/undefined

    hydraulic_power_watts = flow_rate_m3s * pressure_drop_pa
    pump_power_watts = hydraulic_power_watts / pump_efficiency_decimal
    return pump_power_watts / 1000  # Convert to kW

def calculate_energy(pump_power_kw, duration_hours=24):
    """Calculates energy consumption over a period."""
    return pump_power_kw * duration_hours

def calculate_ghg(energy_kwh, emission_factor_kg_co2_kwh=DEFAULT_EMISSION_FACTOR_KG_CO2_KWH):
    """Calculates GHG emissions based on energy consumption."""
    return energy_kwh * emission_factor_kg_co2_kwh

def run_simulation(pipeline: PipelineParams, fluid: FluidProperties, operating: OperatingConditions, calculate_optional: bool) -> Results:
    """Runs the full simulation."""
    if pipeline.diameter_m <= 0 or fluid.viscosity_pas <= 0 or operating.pump_efficiency_decimal <= 0:
        st.error("Invalid input parameters (e.g., zero diameter, viscosity, or efficiency). Cannot simulate.")
        return Results(None, None) # Return empty results

    # 1. Calculate intermediate values
    velocity_ms = calculate_velocity(operating.flow_rate_m3s, pipeline.diameter_m)
    reynolds_number = calculate_reynolds_number(fluid.density_kgm3, velocity_ms, pipeline.diameter_m, fluid.viscosity_pas)
    friction_factor = calculate_friction_factor(reynolds_number, pipeline.roughness_m, pipeline.diameter_m)

    if friction_factor == 0 and reynolds_number >= 2300: # Check if friction factor calculation failed for turbulent
         st.warning("Friction factor calculation resulted in zero for turbulent flow. Results may be inaccurate.")


    # 2. Calculate primary outputs
    pressure_drop_pa = calculate_pressure_drop(pipeline, fluid, velocity_ms, friction_factor)
    pump_power_kw = calculate_pump_power(operating.flow_rate_m3s, pressure_drop_pa, operating.pump_efficiency_decimal)

    # 3. Calculate optional outputs
    energy_kwh_day = None
    ghg_kg_co2e_day = None
    if calculate_optional:
        energy_kwh_day = calculate_energy(pump_power_kw)
        ghg_kg_co2e_day = calculate_ghg(energy_kwh_day)

    return Results(
        pressure_drop_pa=pressure_drop_pa,
        pump_power_kw=pump_power_kw,
        energy_kwh_day=energy_kwh_day,
        ghg_kg_co2e_day=ghg_kg_co2e_day,
        velocity_ms=velocity_ms,
        reynolds_number=reynolds_number,
        friction_factor=friction_factor
    )


# --- Streamlit UI ---
def run_ui(): 
    st.set_page_config(page_title="Pipeline Hydraulic Simulation", layout="wide")
    st.title("💧 Pipeline Hydraulic Simulation")
    st.markdown("Simulate steady-state liquid flow in a pipeline based on the Darcy-Weisbach equation.")

    # --- Sidebar Inputs ---
    st.sidebar.header("⚙️ Inputs")

    # Pipeline Parameters
    st.sidebar.subheader("🔧 Pipeline Parameters")
    length_val = st.sidebar.number_input("Length", min_value=0.1, value=1000.0, step=1.0)
    length_unit = st.sidebar.radio("Length Unit", ('km', 'm'), index=1, horizontal=True)
    length_m = length_val * KM_TO_M if length_unit == 'km' else length_val

    diam_val = st.sidebar.number_input("Internal Diameter", min_value=0.1, value=100.0, step=0.5)
    diam_unit = st.sidebar.radio("Diameter Unit", ('inches', 'mm'), index=1, horizontal=True)
    diameter_m = diam_val * INCHES_TO_M if diam_unit == 'inches' else diam_val * MM_TO_M

    roughness_mm = st.sidebar.number_input("Absolute Roughness (ε)", min_value=0.0, value=0.045, step=0.001, format="%.4f", help="e.g., 0.045 mm for commercial steel")
    roughness_m = roughness_mm * MM_TO_M

    elevation_gain_m = st.sidebar.number_input("Elevation Gain (Inlet to Outlet)", value=0.0, step=5.0, help="Positive for uphill flow (m)")

    pipeline = PipelineParams(length_m, diameter_m, roughness_m, elevation_gain_m)

    # Fluid Properties
    st.sidebar.subheader("🛢️ Fluid Properties")
    density_kgm3 = st.sidebar.number_input("Density (ρ)", min_value=1.0, value=870.0, step=10.0, help="kg/m³ (e.g., 850–950 for crude oil)")
    viscosity_pas = st.sidebar.number_input("Dynamic Viscosity (μ)", min_value=1e-6, value=0.01, step=0.001, format="%.5f", help="Pa·s (e.g., 0.01 - 0.1 for oil, 0.001 for water)") # Changed default and step

    fluid = FluidProperties(density_kgm3, viscosity_pas)

    # Operating Conditions
    st.sidebar.subheader("⚙️ Operating Conditions")
    flow_rate_val = st.sidebar.number_input("Flow Rate", min_value=0.1, value=1000.0, step=100.0)
    flow_rate_unit = st.sidebar.radio("Flow Rate Unit", ('bbl/day', 'm³/s', 'ltrs/min'), index=2, horizontal=True)

    if flow_rate_unit == 'bbl/day':
        flow_rate_m3s = flow_rate_val * BBL_DAY_TO_M3_S
    elif flow_rate_unit == 'ltrs/min':
        flow_rate_m3s = flow_rate_val * LTR_MIN_TO_M3_S
    else: # m³/s
        flow_rate_m3s = flow_rate_val

    pump_efficiency_perc = st.sidebar.slider("Pump Efficiency (%)", min_value=1, max_value=100, value=75, step=1)
    pump_efficiency_decimal = pump_efficiency_perc / 100.0

    operating = OperatingConditions(flow_rate_m3s, pump_efficiency_decimal)

    # Optional Calculations Toggle
    st.sidebar.subheader("📊 Optional Calculations")
    calc_optional = st.sidebar.checkbox("Calculate Energy & GHG Emissions", value=True)

    # --- Main Panel Outputs ---
    st.header("📈 Outputs")

    results = run_simulation(pipeline, fluid, operating, calc_optional)

    if results and results.pressure_drop_pa is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Pressure Drop (ΔP)", value=f"{results.pressure_drop_bar:.3f} bar", delta=f"{results.pressure_drop_pa:.0f} Pa")

        with col2:
            # Check if pump_power_kw is calculated and non-zero before formatting
            power_display = f"{results.pump_power_kw:.2f} kW" if results.pump_power_kw is not None else "N/A"
            st.metric(label="Required Pump Power (P)", value=power_display)


        if calc_optional and results.energy_kwh_day is not None and results.ghg_kg_co2e_day is not None:
            st.subheader("💡 Optional Outputs (Daily)")
            col3, col4 = st.columns(2)
            with col3:
                energy_display = f"{results.energy_kwh_day:.1f} kWh/day" if results.energy_kwh_day is not None else "N/A"
                st.metric(label="Energy Consumption (E)", value=energy_display)
            with col4:
                ghg_display = f"{results.ghg_kg_co2e_day:.1f} kg CO₂e/day" if results.ghg_kg_co2e_day is not None else "N/A"
                st.metric(label="Est. GHG Emissions", value=ghg_display)
                st.caption(f"Based on {DEFAULT_EMISSION_FACTOR_KG_CO2_KWH} kg CO₂e/kWh")

        # Display intermediate calculation details
        with st.expander("View Calculation Details"):
            details_data = {
                "Parameter": ["Fluid Velocity", "Reynolds Number (Re)", "Friction Factor (f)"],
                "Value": [
                    f"{results.velocity_ms:.3f}" if results.velocity_ms is not None else "N/A",
                    f"{results.reynolds_number:.2e}" if results.reynolds_number is not None else "N/A",
                    f"{results.friction_factor:.5f}" if results.friction_factor is not None else "N/A"
                ],
                "Unit": ["m/s", "-", "-"]
            }
            st.table(pd.DataFrame(details_data))
            st.markdown("**Formulas Used:**")
            st.latex(r''' \Delta P = \Delta P_{\text{friction}} + \Delta P_{\text{elevation}} ''')
            st.latex(r''' \Delta P_{\text{friction}} = f \cdot \frac{L}{D} \cdot \frac{\rho v^2}{2} \quad \text{(Darcy-Weisbach)} ''')
            st.latex(r''' \Delta P_{\text{elevation}} = \rho g \Delta H ''')
            st.latex(r''' P_{\text{pump}} = \frac{Q \cdot \Delta P}{\eta_{\text{pump}}} ''')
            st.markdown("Friction factor (f) calculated using Swamee-Jain approximation for turbulent flow (Re > 4000) or f = 64/Re for laminar flow (Re < 2300).")


    else:
        st.info("Adjust the input parameters in the sidebar to run the simulation.")

    # Add explanations or further info if desired
    st.markdown("---")
    st.markdown("### Notes")
    st.markdown("- This simulation assumes steady-state, incompressible, single-phase liquid flow.")
    st.markdown("- Accuracy depends on the friction factor correlation used and input parameter precision.")
    st.markdown("- Results validated against the [Copely Pressure Drop Calculator](https://www.copely.com/discover/tools/pressure-drop-calculator/).")
