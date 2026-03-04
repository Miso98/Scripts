import time
from labjack import ljm

# Initialize LabJack
handle = ljm.openS("T4", "USB", "ANY")
print("LabJack connected.")

# Define DIO pins for Step and Direction
STEP_PIN = 6  # DIO6 connected to Step Pulse
DIR_PIN = 7   # DIO7 connected to Direction

# Configure DIO pins as outputs
ljm.eWriteName(handle, "DIO_INHIBIT", 0xFFFFFFCF)  # Allow changes to DIO6 and DIO7
ljm.eWriteName(handle, "DIO_DIRECTION", 0x000000C0)  # Set DIO6 and DIO7 as outputs (binary 11000000)

# set direction
def set_direction(forward=True):
    direction = 1 if forward else 0
    ljm.eWriteName(handle, f"DIO{DIR_PIN}", direction)
    print(f"Direction set to {'Forward' if forward else 'Reverse'}")

# generate step pulses
def generate_steps(steps, delay=0.001):
    for _ in range(steps):
        ljm.eWriteName(handle, f"DIO{STEP_PIN}", 1)  # Step HIGH
        time.sleep(delay)  # Pulse width
        ljm.eWriteName(handle, f"DIO{STEP_PIN}", 0)  # Step LOW
        time.sleep(delay)  # Delay between pulses

# Main control
try:
    print("Starting motor test...")
    set_direction(forward=True)  # Set forward direction
    generate_steps(1000, delay=0.001)  # Move forward 1000 steps

    time.sleep(1)  # Pause

    set_direction(forward=False)  # Set reverse direction
    generate_steps(1000, delay=0.001)  # Move back 1000 steps

    print("Motor test complete.")
finally:
    print("LabJack disconnected.")
    ljm.close(handle)
