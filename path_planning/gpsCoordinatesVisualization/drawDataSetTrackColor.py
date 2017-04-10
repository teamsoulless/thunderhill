import csv

pi = 3.14159265358979
def RadToDeg (rad):
    return (rad / pi * 180.0)

def speedToClass(speed):
    cl=int(speed/(10*35/80))
    if cl<0:
        cl=0
    if cl>7:
        cl=7
    return cl 

lat0 = 0.0
lon0 = 0.0
with open('coordinates.html', 'w') as htmlfile:
    htmlfile.write("<!DOCTYPE html>\n")
    htmlfile.write("<html>\n")
    htmlfile.write("<head>\n")
    htmlfile.write("<title>Draw a polyline given many coordinates</title>\n")
    htmlfile.write("\n")
    htmlfile.write("<meta name=\"viewport\" content=\"initial-scale=1.0, user-scalable=no\" />\n")
    htmlfile.write("<style type=\"text/css\">\n")
    htmlfile.write("html { height: 100% }\n")
    htmlfile.write("body { height: 100%; margin: 0; padding: 0 }\n")
    htmlfile.write("#map_canvas { height: 100% }\n")
    htmlfile.write("</style>\n")
    htmlfile.write("<script type=\"text/javascript\" src=\"http://maps.googleapis.com/maps/api/js\"></script>\n")
    htmlfile.write("\n")
    htmlfile.write("<script>\n")
    htmlfile.write("   var Colors = [\n")
    htmlfile.write("       \"#00FF00\",\n")
    htmlfile.write("       \"#44FF00\",\n")
    htmlfile.write("       \"#88FF00\",\n")
    htmlfile.write("       \"#AAFF00\",\n")
    htmlfile.write("       \"#FFAA00\",\n")
    htmlfile.write("       \"#FF8800\",\n")
    htmlfile.write("       \"#FF4400\",\n")
    htmlfile.write("       \"#FF0000\",\n")
    htmlfile.write("   ];\n")
    htmlfile.write("    function initialize() {\n")
    htmlfile.write("        var homeLatlng = new google.maps.LatLng(39.53661727905278,-122.33819580078136);\n")
    htmlfile.write("        var myOptions = {\n")
    htmlfile.write("            zoom: 15,\n")
    htmlfile.write("            center: homeLatlng,\n")
    htmlfile.write("            mapTypeId: google.maps.MapTypeId.ROADMAP\n")
    htmlfile.write("        };\n")
    htmlfile.write("        var map = new google.maps.Map(document.getElementById(\"map_canvas\"), myOptions);\n")
    htmlfile.write("\n")
    htmlfile.write("    // create an array of coordinates\n")
    htmlfile.write("    var arrCoords = [\n")

    speedbucket = []
    throttleBrake = []
    with open('output_accel.txt', 'r') as csvfile:
        row = csv.reader(csvfile)
        for fields in row:
            if fields[0] != "path" and fields[0] != "":
                lon1 = RadToDeg(float(fields[3]))
                lat1 = RadToDeg(float(fields[4]))
                if lat0 != lat1 and lon0 != lon1:
                    htmlfile.write("            new google.maps.LatLng({},{}),\n".format(lat1, lon1))
                    lat0 = lat1
                    lon0 = lon1
                    speed = float(fields[18])
                    accel = int(2.0+float(fields[19])*2.0)*15
                    speedbucket.append(speedToClass(speed))
                    throttleBrake.append(accel)
            
    htmlfile.write("        ];\n")
    htmlfile.write("\n")
    htmlfile.write("        // draw the route on the map\n")
    for i in range(len(speedbucket)-1):
        htmlfile.write("        var route = new google.maps.Polyline({\n")
        htmlfile.write("            path: [arrCoords[{}], arrCoords[{}]],\n".format(i,i+1))
        htmlfile.write("            strokeColor: Colors[{}],\n".format(speedbucket[i]))
        htmlfile.write("            strokeOpacity: 1.0,\n")
        htmlfile.write("            strokeWeight: {},\n".format(throttleBrake[i]))
        htmlfile.write("            geodesic: false,\n")
        htmlfile.write("            map: map\n")
        htmlfile.write("        });\n")
    htmlfile.write("    }\n")
    htmlfile.write("\n")
    htmlfile.write("    google.maps.event.addDomListener(window, 'load', initialize);\n")
    htmlfile.write("</script>\n")
    htmlfile.write("</head>\n")
    htmlfile.write("<body>\n")
    htmlfile.write("  <div id=\"map_canvas\"></div>\n")
    htmlfile.write("</body>\n")
    htmlfile.write("</html>\n")

