import xml.etree.ElementTree as ET
import sys
tree = ET.parse('/home/ash/projects/12_regrap_allegro/src/allegro/urdf/allegro_hand_description_right.urdf')
root = tree.getroot()
joints = [j.attrib['name'] for j in root.findall('joint') if j.attrib.get('type') in ['revolute', 'continuous']]
print(joints)
