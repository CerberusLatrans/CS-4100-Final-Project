
# Python program to get a google map 
# image of specified location using 
# Google Static Maps API
  
# importing required modules
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
full_url = url + "center=" + center + "&zoom=" + str(zoom) + "&size=" + size + "&maptype= " + maptype + "&key=" + api_key + "&map_id=" + map_id
print(full_url)
r = requests.get(full_url)
  
with open('map.jpg', 'wb') as file:
    file.write(r.content)

r.close()

# now we repeat this process but with json styling for the clean image
roadOff="feature:road|visibility: off"
labelsTextOff="feature:all|element:labels.text|visibility: off"
labelsIconOff="feature:all|element:labels.icon|visibility: off"
full_url_styled = url + "center=" + center + "&zoom=" + str(zoom) + "&size=" + size + "&maptype= " + maptype + "&map_id=" + map_id + "&style=" + roadOff + "&style=" + labelsTextOff + "&style=" + labelsIconOff + "&key=" + api_key
print(full_url_styled)
rStyled = requests.get(full_url_styled)
with open('styledMap.jpg', 'wb') as file:
    file.write(rStyled.content)

rStyled.close()

# using matpltolib to display the image
plt.figure(figsize=(5, 5))
img=mpimg.imread('map.jpg') #render the terrain map
img=mpimg.imread('styledMap.jpg') #render the styled map
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()

# # wb mode is stand for write binary mode
# f = open('address of the file location ', 'wb')
  
# # r.content gives content,
# # in this case gives image
# f.write(r.content)
  
# # close method of file object
# # save and close the file
# f.close()


# TODO: After we generate image, put into GCP bucket by making GCP API call

