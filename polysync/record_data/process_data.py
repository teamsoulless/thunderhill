# Scirpt to parse txt file containing PlatformMotionMessage and ImageDataMessage 
# into csv file and create directory of files with image data.
# Resuling csv file contains information from PlatformMotionMessage, 
# with timestamps as ids
# Resuling images folder constains files named {timesptamp}.txt, 
# where each timestamp matches a timestamp from PlatformMotionMessage csv file


import json
import csv
import argparse

MOTION_MSG = 'PlatformMotionMessage'
IMAGE_MSG = 'ImageDataMessage'

parser = argparse.ArgumentParser(description='Scirpt to parse txt file containing PlatformMotionMessage and ImageDataMessage into csv file and create directory of files with image data.')

parser.add_argument('-in', '--input', help="Text file to parse", required=True)
parser.add_argument('-c', '--csv', help="Output csv file", required=True)
parser.add_argument('-imd', '--imdir', help="Directory to store files with image data", required=True)
args = parser.parse_args()

txt_file = args.input
csv_file = args.csv
image_dir = args.imdir

print('start processing')

f_lines = []
json_data = []

with open(txt_file) as temp_file:
    f_lines = temp_file.readlines()

for line in f_lines:
    if (line != '\n'):
        try:
            line_json = json.loads(line.strip())
            json_data.append(line_json)
        except ValueError:
            pass

print('JSON is ready')
    
csvfile = open(csv_file, 'w')
fieldnames = ['ts', 'pos_x', 'pos_y', 'pos_z', 
              'orient_1', 'orient_2', 'orient_3', 'orient_4',
              'rr_x', 'rr_y', 'rr_z', 
              'vel_x', 'vel_y', 'vel_z', 
              'acc_x', 'acc_y', 'acc_z', 
              'heading', 'lat', 'lon', 'alt']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

temp_ts = []
matches = []
key = ''
images_data = []
motion_data = []
image_w = 0
image_h = 0

print('Analysing timestamps')
    
for entry in json_data:
    if MOTION_MSG in entry.keys():
        key = MOTION_MSG
        motion_data.append(entry)
    else:
        key = IMAGE_MSG
        images_data.append(entry)
        
    ts = entry[key]['Header']['timestamp']
    if ts not in temp_ts:
        temp_ts.append(ts)
    else:
        matches.append(ts)

print('Writing motion data to csv')
    
for motion_entry in motion_data:
    entry = motion_entry[MOTION_MSG]
    ts = entry['Header']['timestamp']
    if ts in matches:
        pos = entry['position']
        orient = entry['orientation']
        rot_rate = entry['rotation_rate']
        vel = entry['velocity']
        acc = entry['acceleration']
        heading = entry['heading']
        lat = entry['latitude']
        lon = entry['longitude']
        alt = entry['altitude']
        
        #write to csv
        writer.writerow({'ts': ts, 'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2], 
                         'orient_1': orient[0], 'orient_2': orient[1], 'orient_3': orient[2], 'orient_4': orient[3],
                         'rr_x': rot_rate[0], 'rr_y': rot_rate[1], 'rr_z': rot_rate[2], 
                         'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2], 
                         'acc_x': acc[0], 'acc_y': acc[1], 'acc_z': acc[2], 
                         'heading': heading, 'lat': lat, 'lon': lon, 'alt': alt})
        csvfile.flush()
        
print('Writing image data to image directory')
    
for image_entry in images_data:
    entry = image_entry[IMAGE_MSG]
    ts = entry['Header']['timestamp']
    if ts in matches:
        #add to images/ts.txt
        file_name = str(ts) + '.txt'
        img_file = open(image_dir + '/' + file_name, 'w')
        img_file.write(" ".join(entry['data_buffer']))
        image_w = entry['width']
        image_h = entry['height']
        img_file.close()  
        
print('Image processing completed, images have width: ' + str(image_w) + ' and height: ' + str(image_h))

