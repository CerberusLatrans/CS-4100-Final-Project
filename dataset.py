
# Python program to get a google map 
# image of specified location using 
# Google Static Maps API
  
# importing required modules
import requests
  
# Enter your api key here
api_key = "AIzaSyATvdjLWeqJ0svqAXNYmdAmVU7tkaz8434" # Prob want this as a google secret through GCP
  
# url variable store url
url = "https://maps.googleapis.com/maps/api/staticmap?"
  
# center defines the center of the map,
# equidistant from all edges of the map. 
center = "42.3398106,-71.0913604" # TODO: Automate generating coordinates
  
# zoom defines the zoom
# level of the map
zoom = 17

maptype = "terrain"

size = "640x640"

# For styling the map
map_id = "5fd53f1f4a5c6512"
  
# get method of requests module
# return response object
r = requests.get(url + "center=" + center + "&zoom=" +
                   str(zoom) + "&size=" + size + "&maptype=" + maptype + "&key=" +
                             api_key + "&map_id=" + map_id)
  
# wb mode is stand for write binary mode
f = open('address of the file location ', 'wb')
  
# r.content gives content,
# in this case gives image
f.write(r.content)
  
# close method of file object
# save and close the file
f.close()


# TODO: After we generate image, put into GCP bucket by making GCP API call