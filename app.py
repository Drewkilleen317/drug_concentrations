import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configure the page
st.set_page_config(page_title="Drug Concentration Analysis", layout="wide")

# Title
st.title('Drug Concentration Analysis')

# Sidebar controls
with st.sidebar:
    st.header('Parameter Controls')

    # Drug parameters
    dose = st.slider('Dose per injection (mL)', 0.1, 10.0, 5.0, 0.1)
    half_life = st.slider('Elimination Half-life (days)', 1, 14, 7)
    absorption_hours = st.slider('Absorption Half-life (hours)', 1, 48, 17)

    # Schedule parameters
    num_injections = st.slider('Number of Injections', 12, 48, 24)
    total_weeks = st.slider('Total Observation Period (weeks)', 24, 96, 48)

    # Accelerated schedule control (in days)
    accel_days = st.slider('Accelerated Schedule Interval (days)', 1, 7, 6)
    
    # Peak threshold control
    peak_threshold = st.slider('Peak Threshold Percentage', 50, 100, 95) / 100

    # Schedule selection
    st.header('Schedule Selection')
    show_normal = st.checkbox('Normal (7d)', True)
    show_accel = st.checkbox(f'Accelerated ({accel_days}d)', True)

    # Display options
    st.header('Display Options')
    show_raw = st.checkbox('Show Raw Concentrations', True)
    show_avg = st.checkbox('Show Running Averages', True)
    show_peaks = st.checkbox('Show Peak Markers', True)

# Calculate parameters
k_e = np.log(2)/half_life
k_a = np.log(2)/(absorption_hours/24)

def single_dose(t, t_inject, dose, ka, ke):
    if t < t_inject:
        return 0.0
    dt = t - t_inject
    return (dose * ka / (ka - ke)) * (np.exp(-ke*dt) - np.exp(-ka*dt))

def total_conc(t, injection_times, dose, ka, ke):
    return sum(single_dose(t, t_inj, dose, ka, ke) for t_inj in injection_times)

def running_avg(data, window):
    return np.convolve(data, np.ones(window)/window, mode='same')

# Time points
days = total_weeks * 7
t = np.linspace(0, days, 1000)

# Define schedules
schedules = {
    'Normal (7d)': {'interval': 7, 'color': 'blue', 'show': show_normal},
    f'Accelerated ({accel_days}d)': {'interval': accel_days, 'color': 'red', 'show': show_accel}
}

# Calculate concentrations and averages for each schedule
window = int(7 * len(t)/days)  # 7-day window
for schedule in schedules.values():
    injection_times = [schedule['interval'] * i for i in range(num_injections)]
    schedule['conc'] = [total_conc(time, injection_times, dose, k_a, k_e) for time in t]
    schedule['avg'] = running_avg(schedule['conc'], window)
    schedule['peak'] = np.max(schedule['avg'])
    schedule['peak_time'] = t[np.argmax(schedule['avg'])]

    # Calculate time to reach peak threshold
    threshold = peak_threshold * schedule['peak']
    mask = schedule['avg'] >= threshold
    if np.any(mask):
        schedule['time_to_90'] = t[mask][0]  # First time point where avg >= threshold
    else:
        schedule['time_to_90'] = None

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot data for each selected schedule
for name, schedule in schedules.items():
    if schedule['show']:
        if show_raw:
            ax.plot(t, schedule['conc'],
                   color=schedule['color'],
                   alpha=0.3,
                   label=f'{name} Raw')
        if show_avg:
            ax.plot(t, schedule['avg'],
                   color=schedule['color'],
                   linestyle='--',
                   label=f'{name} Avg')
        if show_peaks:
            # Plot peak marker
            ax.scatter(schedule['peak_time'], schedule['peak'],
                      color=schedule['color'],
                      marker='o',
                      label=f'{name} Peak: {schedule["peak"]:.2f}')
            # Plot peak threshold marker
            if schedule['time_to_90'] is not None:
                ax.scatter(schedule['time_to_90'], peak_threshold * schedule['peak'],
                         color=schedule['color'],
                         marker='^',
                         label=f'{name} {peak_threshold*100:.0f}% at {schedule["time_to_90"]:.1f}d')
                # Add horizontal line at peak threshold
                ax.axhline(y=peak_threshold * schedule['peak'],
                          color=schedule['color'],
                          linestyle=':',
                          alpha=0.3)

# Add end of dosing line
ax.axvline(num_injections * 7, color='gray', linestyle='--', label='End of Dosing')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Concentration (mL/L)')
ax.grid(True)
ax.legend()

# Display plot
st.pyplot(fig)

# Display statistics
st.header('Summary Statistics')
col1, col2 = st.columns(2)

for idx, (name, schedule) in enumerate(schedules.items()):
    if schedule['show']:
        with col1 if idx == 0 else col2:
            st.subheader(name)
            st.write(f'Peak Average: {schedule["peak"]:.2f} mL/L')
            st.write(f'Peak Time: {schedule["peak_time"]:.1f} days')
            if schedule['time_to_90'] is not None:
                st.write(f'Time to {peak_threshold*100:.0f}% of Peak: {schedule["time_to_90"]:.1f} days')
                st.write(f'{peak_threshold*100:.0f}% Level: {(peak_threshold * schedule["peak"]):.2f} mL/L')

# Add notes
st.markdown(f"""
---
### Notes:
- Normal Schedule: 168 hours (7 days) between doses
- Accelerated Schedule: {accel_days*24} hours ({accel_days} days) between doses
- Running averages use 7-day windows
- Peak values are based on running averages
- {peak_threshold*100:.0f}% threshold markers (▲) show when running average first reaches {peak_threshold*100:.0f}% of peak
""")

# Add sensitivity analysis
st.header('Sensitivity Analysis')

# Create a function for sensitivity analysis
def run_sensitivity_analysis(base_half_life, base_absorption_hours, half_life_range, absorption_range):
    fig, axs = plt.subplots(len(half_life_range), len(absorption_range), figsize=(15, 10), 
                             sharex=True, sharey=True)
    fig.suptitle('Sensitivity Analysis: Impact of Ke and Ka', fontsize=16)
    
    for i, half_life_mult in enumerate(half_life_range):
        for j, absorption_mult in enumerate(absorption_range):
            # Calculate new parameters
            new_half_life = base_half_life * half_life_mult
            new_absorption_hours = base_absorption_hours * absorption_mult
            
            # Recalculate rate constants
            local_k_e = np.log(2)/new_half_life
            local_k_a = np.log(2)/(new_absorption_hours/24)
            
            # Prepare schedules
            local_schedules = {
                'Normal (7d)': {'interval': 7, 'color': 'blue', 'show': True},
                f'Accelerated ({accel_days}d)': {'interval': accel_days, 'color': 'red', 'show': True}
            }
            
            # Calculate concentrations
            for schedule in local_schedules.values():
                injection_times = [schedule['interval'] * i for i in range(num_injections)]
                schedule['conc'] = [total_conc(time, injection_times, dose, local_k_a, local_k_e) for time in t]
                schedule['avg'] = running_avg(schedule['conc'], window)
                schedule['peak'] = np.max(schedule['avg'])
            
            # Plot on specific subplot
            ax = axs[i, j]
            
            # Plot Normal Schedule
            normal_schedule = local_schedules['Normal (7d)']
            ax.plot(t, normal_schedule['avg'], color='blue', label='Normal (7d)')
            
            # Plot Accelerated Schedule
            accel_schedule = local_schedules[f'Accelerated ({accel_days}d)']
            ax.plot(t, accel_schedule['avg'], color='red', label=f'Accelerated ({accel_days}d)')
            
            # Annotate subplot
            ax.set_title(f'Half-life: {half_life_mult:.2f}x\nAbsorption: {absorption_mult:.2f}x')
            
            if i == len(half_life_range) - 1:
                ax.set_xlabel('Time (days)')
            if j == 0:
                ax.set_ylabel('Concentration (mL/L)')
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# Add sensitivity analysis controls
st.subheader('Sensitivity Analysis Parameters')
col1, col2 = st.columns(2)

with col1:
    half_life_options = st.multiselect(
        'Half-life Multipliers', 
        [0.5, 0.75, 1.0, 1.25, 1.5], 
        default=[0.5, 1.0, 1.5]
    )

with col2:
    absorption_options = st.multiselect(
        'Absorption Rate Multipliers', 
        [0.5, 0.75, 1.0, 1.25, 1.5], 
        default=[0.5, 1.0, 1.5]
    )

# Button to run sensitivity analysis
if st.button('Run Sensitivity Analysis'):
    run_sensitivity_analysis(
        base_half_life=half_life, 
        base_absorption_hours=absorption_hours, 
        half_life_range=half_life_options, 
        absorption_range=absorption_options
    )