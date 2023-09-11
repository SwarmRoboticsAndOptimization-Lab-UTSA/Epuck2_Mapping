# Epuck2_Mapping
Fall 2023 Epuck swarm mapping

Current working files are:

- Run_AprilTag_Det.py For a simple april tag detection that shows the number of the april tag detected.
- Single_Robot_controll_w_April tags.py This code controls a single robot and makes it move to a desired location based on the april tag detected.
    - Robot needs to have an april tag on top of it, and be under the camera field of view.
    - The robot must be turned on and the computer must be connected to Linksis03735 network
- Modify Camera Settings.py is used to change values that affect the image the camera captures.
- apriltag-gen.ipynb is used to create more april tags.