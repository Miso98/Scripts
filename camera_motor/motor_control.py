import time
from labjack import ljm

# Initialize LabJack T4
handle = ljm.openS("T4", "ANY", "ANY")

# Pin configuration
STEP_PIN = 6  # DIO6 connected to Step Pulse
DIR_PIN = 7   # DIO7 connected to Direction
LIMIT_SWITCH_PIN = "AIN0"  # AIN0 connected to the limit switch

# Configure DIO pins as outputs
ljm.eWriteName(handle, "DIO_INHIBIT", 0xFFFFFFCF)  # Allow changes to DIO6 and DIO7
ljm.eWriteName(handle, "DIO_DIRECTION", 0x000000C0)  # Set DIO6 and DIO7 as outputs (binary 11000000)

# Global variables
current_position = 0  # Tracks the current position of the actuator
steps_per_mm = 100  # Example: Adjust this based on calibration

def move_stepper(steps, direction):
    
    # Move the stepper motor a specified number of steps in a given direction.
    # :param steps: Number of steps to move.
    # :param direction: 0 for clockwise, 1 for counterclockwise.
    
    global current_position
    # Set direction
    ljm.eWriteName(handle, f"DIO{DIR_PIN}", direction)
    
    # Generate step pulses
    for _ in range(steps):
        ljm.eWriteName(handle, f"DIO{STEP_PIN}", 1)  # Step high
        time.sleep(0.0005)  # Pulse width (adjust as needed)
        ljm.eWriteName(handle, f"DIO{STEP_PIN}", 0)  # Step low
        time.sleep(0.0005)  # Delay between steps (adjust for speed)
    
    # Update current position
    current_position += steps if direction else -steps

def read_limit_switch():
   
    #Read the limit switch state from AIN0.
    #return True if the switch is triggered (voltage near 0V), False otherwise.
   
    voltage = ljm.eReadName(handle, LIMIT_SWITCH_PIN)
    return voltage < 0.5  # Switch is triggered if voltage is below 0.5V

def home_actuator():
    #Home actuator by moving it toward the limit switch   
    global current_position
    print("Homing actuator...")
    while not read_limit_switch():
        move_stepper(1, 0)  # Move 1 step in the backward direction (adjust direction as needed)
        time.sleep(0.01)  # Small delay to avoid overshooting
    print("Home position reached.")
    current_position = 0  # Reset current position to 0


def main():
    try:
        # Home the actuator
        home_actuator()
        
        # Calibrate the actuator (need to fill in)
        
        
        while True:
            # Prompt user for input
            user_input = input("Enter distance to move in mm (positive for forward, negative for backward, 'q' to quit): ")
            
            if user_input.lower() == 'q':
                break
            
            try:
                distance_mm = float(user_input)
                direction = 1 if distance_mm > 0 else 0  # 1 for forward, 0 for backward
                steps = int(abs(distance_mm) * steps_per_mm)
                
                # Move the stepper motor
                move_stepper(steps, direction)
                print(f"Moved {distance_mm} mm {'forward' if direction else 'backward'}. Current position: {current_position} steps")
            
            except ValueError:
                print("Invalid input. Please enter a valid distance in mm.") 
    
    except KeyboardInterrupt:
        print("Program terminated by user.")
    
    finally:
        # Clean up
        ljm.close(handle)

if __name__ == "__main__":
    main()