#!/usr/bin/env python
import socket
import time
from threading import Thread
import logging
import numpy as np
import cv2
import tensorflow as tf
from obj_det_utils.utils import *
from pupil_apriltags import Detector
import copy
import platform
import subprocess

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
NUM_ROBOTS = 6 # Set this value to the number of robots to which connect
# Fill the list with the IP addresses of the robots to which connect
# addr = ["192.168.1.100","192.168.1.101","192.168.1.102","192.168.1.103","192.168.1.104","192.168.1.105","192.168.1.106","192.168.1.107","192.168.1.108"]
addr = ["192.168.1.100","192.168.1.101","192.168.1.102","192.168.1.106","192.168.1.107","192.168.1.108"]

# Fill the list with the IDs of the robots to which connect (the sequence must reflect the one of the IP addresses)
# robot_id = ['5790','5792','5462','5739','5664','5662','5387','5788','5663']
robot_id = ['5790','5792','5462','5387','5788','5663']

#Robots Dictionary
robot_dic = {}
for id in robot_id:#Initiate all values so that robot does not fail if there is no detection
    robot_dic[id] = [0,360,1000]  # First val is heading, then desired heading, then dist

command_dict = {}
taken_locations = {}

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
continue_command = [True for _ in range(NUM_ROBOTS)]# Array to store flags if a robot should continue receiving commands or not.
distance_to_goal_list = [None for _ in range(NUM_ROBOTS)] #Array to store distances

expected_recv_packets = 0

######################
## OBJECT DETECTION ##
######################

# Initialize OpenCV video capture
if platform.system() == 'Windows':
    # Do something specific for Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_SETTINGS,1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    #Robot desired Locations #FOR NOW Locations must be in range x (250,825) and y (25, 550)

    desired_location = [[282,96],[414,96],[546,96],[813,96],[480,120],[480,205],[480,290],[480,375],[480,548]]
    display_locations = desired_location.copy()


elif platform.system() == 'Linux':
    lin_commands = [
        "v4l2-ctl --set-ctrl=auto_exposure=1",
        "v4l2-ctl --device=/dev/video0 --set-ctrl=exposure_time_absolute=120",
        "v4l2-ctl --device=/dev/video0 --set-ctrl=contrast=25",
        "v4l2-ctl --device=/dev/video0 --set-ctrl=brightness=0"
        ]

    for lin_command in lin_commands:
        try:
            subprocess.check_call(lin_command, shell=True)
        except subprocess.CalledProcessError:
            print(f"Error executing the command: {lin_command}")
    # Do something specific for Linux
    cap = cv2.VideoCapture(0)

    desired_location = [[160,66],[253,66],[345,66],[438,66],[515,66],[345,148],[345,243],[345,327],[345,412]]
    display_locations = desired_location.copy()

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
    ########################
    ## April Tag Detector ##
    ########################
    at_detector = Detector(
        families="tagCustom48h12",
        nthreads=1,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    global command
    global proximity
    global gyro
    global refresh
    global led_state
    global num_packets
    global continue_command
    trials = 0
    sensors = bytearray([0] * SENSORS_PACKET_SIZE)
    socket_error = 0
    closest_location = None
        
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
        print(continue_command)
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

        if not continue_command[client_index]:
            des_speed_right = 0
            des_speed_left = 0 #500 FORWARD 400 TURN LEFT WHILE MOVING FORWARD
            left_motor_LSB, left_motor_MSB = get_motor_bytes(des_speed_left)
            right_motor_LSB, right_motor_MSB = get_motor_bytes(des_speed_right)
                    
            command[i][3] = left_motor_LSB      # left motor LSB
            command[i][4] = left_motor_MSB     # left motor MSB
            command[i][5] = right_motor_LSB    # right motor LSB
            command[i][6] = right_motor_MSB    # right motor MSB
                
            send(client_sock, command[client_index], COMMAND_PACKET_SIZE)
            # print(client_index, continue_command[client_index])
            client_sock.close()
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

                # Desired speeds
                des_speed_right = 0
                des_speed_left = 0 #500 FORWARD 400 TURN LEFT WHILE MOVING FORWARD
                des_speeds = {}

                ret, frame = cap.read()
                if not ret:
                    break
                debug_image = copy.deepcopy(frame)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tags = at_detector.detect(
                    image,
                    estimate_tag_pose=False,
                    camera_params=None,
                    tag_size=None,
                )

                for tag in tags:
                    tag_family = tag.tag_family
                    tag_id = tag.tag_id
                    center = tag.center
                    corners = tag.corners
                    center = (int(center[0]), int(center[1]))
                    corner_01 = (int(corners[0][0]), int(corners[0][1]))
                    corner_02 = (int(corners[1][0]), int(corners[1][1]))
                    # corner_03 = (int(corners[2][0]), int(corners[2][1]))
                    # corner_04 = (int(corners[3][0]), int(corners[3][1]))
                    min_distance = float('inf')

                    mid = midpoint(corner_01,corner_02)

                    cv2.line(debug_image, (center[0], center[1]),(mid[0], mid[1]), (255, 255, 0), 2)

                    heading = calculate_heading(center,mid)

                    distances = [calculate_distance(mid[0],mid[1],des_loc[0],des_loc[1]) for des_loc in desired_location]
                    if distances:
                        min_distance = distances.index(min(distances)) #Get the index of the smallest distance.
                        taken_locations[str(tag_id)] = desired_location[min_distance] #Use index of the smallest distance to update robot desired location
                        desired_location.pop(min_distance)
                    
                    try:
                        dist = calculate_distance(mid[0],mid[1],taken_locations[str(tag_id)][0],taken_locations[str(tag_id)][1])
                        desired_heading = calculate_heading(center,taken_locations[str(tag_id)])
                        robot_dic[str(tag_id)] = [heading,desired_heading, dist]
                    except:
                        pass
                    # print(tag_id)
                
                if robot_dic:
                    c_ind = 0
                    # print(robot_dic)
                    for id in robot_id:
                        if robot_id.index(id) == client_index:
                            distance_to_goal_list[client_index] = robot_dic[id][2]
                            rotation_direction = calculate_rotation_direction(robot_dic[id][0],robot_dic[id][1])

                            if rotation_direction == "no rotation":
                                des_speed_left = 150
                                des_speed_right = 150

                            elif rotation_direction == "left":
                                des_speed_left = -150
                                des_speed_right = 150
                            
                            elif rotation_direction == "right":
                                des_speed_left = 150
                                des_speed_right = -150

                            # distance_to_goal = robot_dic[id][2]
                            if distance_to_goal_list[client_index] <= 10:
                                des_speed_left = 0
                                des_speed_right = 0
                                continue_command[client_index] = False  # Set flag to stop sending commands to this robot.

                            command_dict[str(c_ind)] = [des_speed_left, des_speed_right,id]
                            c_ind +=1
                                
                for i in display_locations:
                    cv2.circle(debug_image, i, 10, (0,255,255), -1) #Draw circle goal location
                
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    des_speed_right = 0
                    des_speed_left = 0 #500 FORWARD 400 TURN LEFT WHILE MOVING FORWARD
                    break

                # print(command_dict)
                for i in command_dict:
                    i = int(i)
                    des_speed_left = command_dict[str(i)][0]
                    des_speed_right = command_dict[str(i)][1]
                    #Temporal
                    # des_speed_left = 0
                    # des_speed_right = 0

                    left_motor_LSB, left_motor_MSB = get_motor_bytes(des_speed_left)
                    right_motor_LSB, right_motor_MSB = get_motor_bytes(des_speed_right)
                    
                    command[i][3] = left_motor_LSB      # left motor LSB
                    command[i][4] = left_motor_MSB     # left motor MSB
                    command[i][5] = right_motor_LSB    # right motor LSB
                    command[i][6] = right_motor_MSB    # right motor MSB
                
                    num_packets[client_index] += 1
                # print(num_packets)
            
                        
            elif header == bytearray([3]): # Empty ack
                print(client_addr + " received an empty packet\r\n")
            else:
                print(client_addr + ": unexpected packet\r\n")
                
            expected_recv_packets -= 1
            cv2.imshow("img", debug_image)

    client_sock.close()


# Start a dedicated thread for each robot.
for x in range(NUM_ROBOTS):
    #print(x)
    new_client(x, sock[x], addr[x])
    time.sleep(1)

# Main loop: print some information about all the robots every 2 seconds.
# while True:
# 	time.sleep(2)
# 	print("#\tID\tIP\t\tSensor\tActuator\tN.packets")
# 	for x in range(NUM_ROBOTS):
# 		print(str(x) + "\t" + robot_id[x] + "\t" + addr[x] + "\t" + str(proximity[x][0]) + "\t" + str(led_state[x]) + "\t\t" + str(num_packets[x]))
# 		num_packets[x] = 0
# 	print("\r\n")


# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()