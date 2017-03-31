import csv

pi = 3.14159265358979
def RadToDeg (rad):
    return (rad / pi * 180.0)

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

    with open('output.txt', 'r') as csvfile:
        latlons = csv.reader(csvfile)
        for latlon in latlons:
            if latlon[0] != "path" and latlon[0] != "":
                lon1 = RadToDeg(float(latlon[2]))
                lat1 = RadToDeg(float(latlon[3]))
                if lat0 != lat1 and lon0 != lon1:
                    htmlfile.write("            new google.maps.LatLng({},{}),\n".format(lat1, lon1))
                    lat0 = lat1
                    lon0 = lon1
    htmlfile.write("        ];\n")
    htmlfile.write("\n")
    htmlfile.write("        // draw the route on the map\n")
    htmlfile.write("        var route = new google.maps.Polyline({\n")
    htmlfile.write("            path: arrCoords,\n")
    htmlfile.write("            strokeColor: \"#00FF00\",\n")
    htmlfile.write("            strokeOpacity: 1.0,\n")
    htmlfile.write("            strokeWeight: 4,\n")
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

