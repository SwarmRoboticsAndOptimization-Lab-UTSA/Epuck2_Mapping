import os
import xml.etree.ElementTree as ET

def modify_xml_path(directory, new_path_prefix):
    # Loop through each XML file in the directory
    for xml_file in os.listdir(directory):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(directory, xml_file)

            # Parse the XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Find the filename tag and modify its text
            filename_tag = root.find("path")
            if filename_tag is not None:
                # Use the XML file's name (without the .xml extension) as the image name
                image_name = os.path.splitext(xml_file)[0] + ".jpg"  # assuming the images are .jpg
                filename_tag.text = os.path.join(new_path_prefix, image_name)

            # Write the modified XML back to the file
            tree.write(xml_path)

# Use it like this
directory_path = "/home/elusis/Documents/Fall2023/Epuck2_Mapping/epuck_images"
new_path_prefix = "/home/elusis/Documents/Fall2023/Epuck2_Mapping/epuck_images"  # adjust this to your desired prefix
modify_xml_path(directory_path, new_path_prefix)
