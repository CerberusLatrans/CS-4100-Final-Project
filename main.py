# creates n uniformly random samples within radius r from the location
def get_coords(location, n, radius):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(location) +'?format=json'
    response = requests.get(url).json()
    anchor_lat = float(response[0]["lat"]) #y
    anchor_lon = float(response[0]["lon"]) #x

    #convert miles to lat degrees
    r = radius / 69

    coordinates = []

    for _ in range(n):
        #latitude can be +- radius from the anchor lat
        lat = anchor_lat + random.uniform(-r, r)

        # from circle eqn
        delta_lon = math.sqrt(r**2 - (lat - anchor_lat)**2)
        lon = anchor_lon + random.uniform(-delta_lon, delta_lon)

        zoom = random.randint(17, 20)
        coordinates.append((lat, lon, zoom))

    #print(coordinates)
    return coordinates


def upload_to_cloud(c):
    pass

locations = ["New York", "Boston"]
coords = []
for l in locations:
    coords.extend(get_coords(l, n=10, radius=5))
for c in coords:
    upload_to_cloud(c)