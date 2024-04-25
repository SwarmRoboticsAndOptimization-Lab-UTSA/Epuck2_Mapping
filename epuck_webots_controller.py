import math, time
from controller import Robot, DistanceSensor, Motor, GPS

def run_robot(robot):
    looking_heading = True
    in_place = False
    timestep = int(robot.getBasicTimeStep())
    max_speed = 6.28
    gyro = robot.getDevice('gyro')
    gyro.enable(timestep)
    leftMotor = robot.getDevice('left wheel motor')
    rightMotor = robot.getDevice('right wheel motor')
    prox_0 = robot.getDevice('ps0')
    prox_1 = robot.getDevice('ps1')
    prox_2 = robot.getDevice('ps2')
    prox_3 = robot.getDevice('ps3')
    prox_4 = robot.getDevice('ps4')
    prox_5 = robot.getDevice('ps5')
    prox_6 = robot.getDevice('ps6')
    prox_7 = robot.getDevice('ps7')
    prox_0.enable(timestep)
    prox_1.enable(timestep)
    prox_2.enable(timestep)
    prox_3.enable(timestep)
    prox_4.enable(timestep)
    prox_5.enable(timestep)
    prox_6.enable(timestep)
    prox_7.enable(timestep)

    prox_values_list = []
    step_count = 0

    # Specify the file path
    file_path = '/home/swarmlab/Documents/E_Franco_Robot_Work/Epuck2/proximity_data.txt'


    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))

    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    
    def move_forward(robot):
        leftMotor.setVelocity(max_speed * 0.25)
        rightMotor.setVelocity(max_speed * 0.25)

    def rotate_left(robot):
        leftMotor.setVelocity(-max_speed * 0.25)
        rightMotor.setVelocity(max_speed * 0.25)

    def rotate_right(robot):
        leftMotor.setVelocity(max_speed * 0.25)
        rightMotor.setVelocity(-max_speed * 0.25)

    def stop(robot):
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)
   
    while robot.step(timestep) != -1:   
        step_count += 1

        # Get the values from each proximity sensor
        prox_values = [prox_0.getValue(),prox_1.getValue(),prox_2.getValue(),prox_3.getValue(),prox_4.getValue(),prox_5.getValue(),prox_6.getValue(),prox_7.getValue()]
        # Store the list of values in prox_values_list
        prox_values_list.append(prox_values)

        print("Prox", prox_values_list)

        # Save the data to the file every 100 steps
        if step_count % 100 == 0:
            # Open the file in append mode to update the existing data
            with open(file_path, 'a') as file:
                for values in prox_values_list:
                    file.write(f"{values}\n")
            # Clear the list after saving to manage memory usage
            prox_values_list.clear()

        #print(gyro_value)
        #leftMotor.setVelocity(0)
        #rightMotor.setVelocity(0)
        
if __name__ == '__main__':
    robot = Robot()
    run_robot(robot)