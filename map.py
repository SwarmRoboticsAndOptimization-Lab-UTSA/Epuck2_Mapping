#!/usr/bin/env python

import socket
import sys
import time
from threading import Thread
import logging
import json
import numpy as np
import random, math
import matplotlib.pyplot as plt

from filelock import FileLock

###############
## CONSTANTS ##
###############
COMMAND_PACKET_SIZE = 21
HEADER_PACKET_SIZE = 1
SENSORS_PACKET_SIZE = 104
IMAGE_PACKET_SIZE = 38400 # Max buffer size = 160x120x2
MAX_NUM_CONN_TRIALS = 5
SENS_THRESHOLD = 200
TCP_PORT = 1000 # This is fixed.

##############################
## TO BE FILLED BY THE USER ##
##############################
NUM_ROBOTS = 1 # Set this value to the number of robots to which connect
# Fill the list with the IP addresses of the robots to which connect
addr = ["192.168.1.102"]

# addr = ["192.168.1.238", "192.168.1.126", "192.168.1.127", "192.168.1.128", "192.168.1.129", "192.168.1.130", "192.168.1.131", "192.168.1.132", "192.168.1.133", "192.168.1.134"]
# Fill the list with the IDs of the robots to which connect (the sequence must reflect the one of the IP addresses)
robot_id = ["5792"]

# robot_id = ["4539", "4889", "4787", "4946", "4995", "4523", "4516", "4703", "4704", "4700"]

###############
## VARIABLES ##
###############
sock = [None] * NUM_ROBOTS
header = bytearray([0] * 1)
image = bytearray([0] * IMAGE_PACKET_SIZE)
proximity = [[0 for x in range(8)] for y in range(NUM_ROBOTS)] # Matrix containing the proximity values for all robots.
gyro = [[0 for x in range(6)] for y in range(NUM_ROBOTS)] # Matrix containing the gyro values for all robots.
command = [bytearray([0] * COMMAND_PACKET_SIZE) for y in range(NUM_ROBOTS)] # Matrix containing the commands sent by all the robots.
refresh = [0 for x in range(NUM_ROBOTS)]
led_state = [0 for x in range(NUM_ROBOTS)]
num_packets = [0 for x in range(NUM_ROBOTS)]

expected_recv_packets = 0


#############################
## COMMUNICATION FUNCTIONS ##
#############################
def send(s, msg, msg_len):
	totalsent = 0
	while totalsent < msg_len:
		sent = s.send(msg[totalsent:])
		if sent == 0:
			raise RuntimeError("Send error")
		totalsent = totalsent + sent

def receive(s, msg_len):
	chunks = []
	bytes_recd = 0
	while bytes_recd < msg_len:
		chunk = s.recv(min(msg_len - bytes_recd, 2048))
		if chunk == b'':
			raise RuntimeError("Receive error")
		chunks.append(chunk)
		bytes_recd = bytes_recd + len(chunk)
	return b''.join(chunks)		

####################
## CONTROL THREAD ##
####################
# One thread is created for each robot.
# The thread is responsible of: initiating and maintaining the communication; interpreting the data received and controlling the robot based on these data.
# Some global variables are used. In the thread these variables are accessed based on the "index" passed as argument.
def new_client(client_index, client_sock, client_addr):
    
    global command
    global proximity
    global gyro
    global refresh
    global led_state
    global num_packets
    global correct_heading
    trials = 0
    sensors = bytearray([0] * SENSORS_PACKET_SIZE)
    socket_error = 0
    all_ir_readings = []
    robot_position = [5.0,5.0]
    map_width = 100
    map_height = 100
    grid_resolution = 0.1
    max_sensor_distance = 0.06  # 6 cm in meters

    # Initialize the occupancy grid
    grid_rows = int(map_height  / grid_resolution)
    grid_cols = int(map_width  / grid_resolution)               
    occupancy_grid = np.zeros((grid_rows, grid_cols))
    
    def get_motor_bytes(speed):
        LSB = speed & 0xFF
        MSB = (speed >> 8) & 0xFF
        return LSB, MSB

    # Init the array containing the commands to be sent to the robot.
    command[client_index][0] = 0x80;	# Packet id for settings actuators
    command[client_index][1] = 2;		# Request: only sensors enabled
    command[client_index][2] = 0;		# Settings: set motors speed
    command[client_index][3] = 0		# left motor LSB
    command[client_index][4] = 0		# left motor MSB
    command[client_index][5] = 0		# right motor LSB
    command[client_index][6] = 0		# right motor MSB
    command[client_index][7] = 0x00;	# lEDs
    command[client_index][8] = 0;		# LED2 red
    command[client_index][9] = 0;		# LED2 green
    command[client_index][10] = 0;		# LED2 blue
    command[client_index][11] = 0;		# LED4 red	
    command[client_index][12] = 0;		# LED4 green
    command[client_index][13] = 0;		# LED4 blue
    command[client_index][14] = 0;		# LED6 red
    command[client_index][15] = 0;		# LED6 green
    command[client_index][16] = 0;		# LED6 blue
    command[client_index][17] = 0;		# LED8 red
    command[client_index][18] = 0;		# LED8 green
    command[client_index][19] = 0;		# LED8 blue
    command[client_index][20] = 0;		# speaker

    # Init the connection. In case of errors, try again for a while and eventually give up in case the connection cannot be accomplished.
    print("Try to connect to " + client_addr + ":" + str(TCP_PORT) + " (TCP)")			
    while trials < MAX_NUM_CONN_TRIALS:
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_sock.settimeout(10) # non-blocking socket
        try:
            client_sock.connect((client_addr, TCP_PORT))
        except socket.timeout as err:
            client_sock.close()
            logging.error("Error from " + client_addr + ":")
            logging.error(err)
            trials += 1
            continue
        except socket.error as err:
            client_sock.close()
            logging.error("Error from " + client_addr + ":")
            logging.error(err)
            trials += 1
            continue
        except Exception as err:
            client_sock.close()
            logging.error("Error from " + client_addr + ":")
            logging.error(err)
            trials += 1
            continue
        break
            
    if trials == MAX_NUM_CONN_TRIALS:
        print("Can't connect to " + client_addr)
        return
        
    print("Connected to " + client_addr)
    print("\r\n")

    while True:
        # If there was some errors in sending or receiving then try to close the connection and reconnect.
        if socket_error == 1:
            socket_error = 0
            trials = 0
            while trials < MAX_NUM_CONN_TRIALS:
                client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    client_sock.connect((client_addr, TCP_PORT))
                except socket.timeout as err:
                    client_sock.close()
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    trials += 1
                    continue
                except socket.error as err:
                    client_sock.close()
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    trials += 1
                    continue
                except Exception as err:
                    client_sock.close()
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    trials += 1
                    continue					
                break
            if trials == MAX_NUM_CONN_TRIALS:
                print("Can't reconnect to " + client_addr)
                break
        
        # Send a command to the robot.
        try:
            send(client_sock, command[client_index], COMMAND_PACKET_SIZE)
        except socket.timeout as err:
            logging.error("Error from " + client_addr + ":")
            logging.error(err)
            socket_error = 1
            continue
        except socket.error as err:
            logging.error("Error from " + client_addr + ":")
            logging.error(err)
            socket_error = 1
            continue
        except Exception as err:
            logging.error("Error from " + client_addr + ":")
            logging.error(err)
            socket_error = 1
            continue			

        # Set the number of expected packets to receive based on the request done.
        #expected_recv_packets = 2 # Camera and sensors.
        expected_recv_packets = 1 # Only sensors.
        
        while(expected_recv_packets > 0):
            # Get the first byte to distinguish the content of the packet.
            try:
                header = receive(client_sock, HEADER_PACKET_SIZE)
                #print("header=" + str(header[0]))
            except socket.timeout as err:
                logging.error("Error from " + client_addr + ":")
                logging.error(err)				
                socket_error = 1
                break
            except socket.error as err:
                logging.error("Error from " + client_addr + ":")
                logging.error(err)
                socket_error = 1
                break
            except Exception as err:
                logging.error("Error from " + client_addr + ":")
                logging.error(err)
                socket_error = 1
                break				
            
            if header == bytearray([1]): # Get a QQVGA image
                try:
                    image = receive(client_sock, IMAGE_PACKET_SIZE)
                except socket.timeout as err:
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    socket_error = 1
                    break
                except socket.error as err:
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    socket_error = 1
                    break
                except Exception as err:
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    socket_error = 1
                    break
                    
            elif header == bytearray([2]): # Get sensors data
               
                try:
                    sensor = receive(client_sock, SENSORS_PACKET_SIZE)
                except socket.timeout as err:
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    socket_error = 1
                    break
                except socket.error as err:
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    socket_error = 1
                    break
                except Exception as err:
                    logging.error("Error from " + client_addr + ":")
                    logging.error(err)
                    socket_error = 1
                    break					
                proximity[client_index][0] = sensor[37] + sensor[38]*256
                proximity[client_index][1] = sensor[39] + sensor[40]*256
                proximity[client_index][2] = sensor[41] + sensor[42]*256
                proximity[client_index][3] = sensor[43] + sensor[44]*256
                proximity[client_index][4] = sensor[45] + sensor[46]*256
                proximity[client_index][5] = sensor[47] + sensor[48]*256
                proximity[client_index][6] = sensor[49] + sensor[50]*256
                proximity[client_index][7] = sensor[51] + sensor[52]*256

                gyro[client_index][0] = sensor[18]
                gyro[client_index][1] = sensor[19]
                gyro[client_index][2] = sensor[20]
                gyro[client_index][3] = sensor[21]
                gyro[client_index][4] = sensor[22]
                gyro[client_index][5] = sensor[23]

                # Here goes your controller.
                # Desired speeds
                des_speed_right = 0
                des_speed_left = 0 #500 FORWARD 400 TURN LEFT WHILE MOVING FORWARD
               
                robot_speed = 0.05
                update_interval = 0.1

                # Get motor bytes
                right_motor_LSB, right_motor_MSB = get_motor_bytes(des_speed_right)
                left_motor_LSB, left_motor_MSB = get_motor_bytes(des_speed_left)
                command[client_index][3] = left_motor_LSB		# left motor LSB
                command[client_index][4] = left_motor_MSB		# left motor MSB
                command[client_index][5] = right_motor_LSB		# right motor LSB
                command[client_index][6] = right_motor_MSB		# right motor MSB

                ir_readings = proximity[0]
                for angle, ir_reading in enumerate(ir_readings):
                    normalized_distance = (ir_reading / 1000) * max_sensor_distance

                    # Calculate position in the direction of the sensor reading
                    sensor_x = robot_position[0] + normalized_distance * np.cos(np.radians(angle * 45))
                    sensor_y = robot_position[1] + normalized_distance * np.sin(np.radians(angle * 45))

                    plt.scatter(sensor_x, sensor_y, color='lime', s=10)  # Bright color and bigger size

                    # Convert position to grid coordinates
                    grid_row = int(sensor_y / grid_resolution)
                    grid_col = int(sensor_x / grid_resolution)

                    robot_position[0] += robot_speed * 1
                    robot_position[1] += robot_speed * 1

                    # Update occupancy grid
                    if 0 <= grid_row < grid_rows and 0 <= grid_col < grid_cols:
                        occupancy_grid[grid_row][grid_col] = 1.0 if normalized_distance < 0.03 else 0.3

                all_ir_readings.append(ir_readings)
                #import csv

                # with open('ir_readings.csv', 'a', newline='') as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerow(ir_readings)


                #print(len(all_ir_readings))      
                 # Visualize the occupancy grid
                plt.imshow(occupancy_grid, cmap='gray', origin='lower', extent=(0, map_width, 0, map_height))
                plt.colorbar()
                plt.scatter(robot_position[0], robot_position[1], color='red', label='Robot Position')
                plt.xlabel('X (meters)')
                plt.ylabel('Y (meters)')
                plt.title('Occupancy Grid Map with IR Readings')
                plt.legend()
                plt.show()               

                num_packets[client_index] += 1
                # print(num_packets)
                        
            elif header == bytearray([3]): # Empty ack
                print(client_addr + " received an empty packet\r\n")
            else:
                print(client_addr + ": unexpected packet\r\n")
                
            expected_recv_packets -= 1

    client_sock.close()

# Start a dedicated thread for each robot.
threads = []
for x in range(NUM_ROBOTS):
	t = Thread(target=new_client, args=(x, sock[x], addr[x]))
	t.start()
	threads.append(t)
	time.sleep(1)

# Join all threads.
#for t in threads:
#    t.join()

# Main loop: print some information about all the robots every 2 seconds.
while True:
	time.sleep(2)
	print("#\tID\tIP\t\tSensor\tActuator\tN.packets")
	for x in range(NUM_ROBOTS):
		print(str(x) + "\t" + robot_id[x] + "\t" + addr[x] + "\t" + str(proximity[x][0]) + "\t" + str(led_state[x]) + "\t\t" + str(num_packets[x]))
		num_packets[x] = 0
	print("\r\n")

