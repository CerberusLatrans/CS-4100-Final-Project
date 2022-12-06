import heapq
import cv2
from matplotlib import pyplot as plt

class Node():
    def __init__(self, coordinate, path):
        self.coordinate = coordinate
        self.path = path
    def __lt__(self, other):
        return len(self.path) < len(other.path)
    

def a_star(map, start, destination, heuristic):

    curNode = (0, Node(start, []))
    frontier = [curNode]
    explored = []

    while len(frontier) > 0: 
        #print([x[0] for x in frontier])
        nodeCost, curNode = heapq.heappop(frontier)
        #print(len(curNode.path), curNode.coordinate)
        #print(heuristic(curNode.coordinate, destination))
        if curNode.coordinate not in explored:
            #print(curNode.coordinate)
            explored.append(curNode.coordinate)
            if curNode.coordinate == destination:
                print("FOUND")
                return curNode.path, True
            else: 
                for neighbor in getSuccessors(curNode, map):
                #each neighbor is a (successor,action, stepCost)
                #need to create new Action as a copy!!! otherwise unwanted side effect
                    heapq.heappush(frontier, (len(neighbor.path) + heuristic(neighbor.coordinate, destination), neighbor))

    
    print("PATH NOT FOUND")
    return curNode.path, False

def getSuccessors(curNode, map):
    # note x is row y is column
    x, y = curNode.coordinate
    successors = []
    listOfCoords = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]
    for coord in listOfCoords: 
        curX, curY = coord
        if curX >= 0 and curY >= 0 and curX < len(map) and curY < len(map[0]) and map[curX][curY]:  #it's black
            successors.append(Node(coord, curNode.path + [coord]))
            print(coord)
    return successors


def manhattanHeuristic(coord, destination): 
    coordX, coordY = coord
    destX, destY = destination
    return abs(destX - coordX) + abs(destY - coordY)

def euclideanHeuristic(coord, destination): 
    coordX, coordY = coord
    destX, destY = destination
    return ((coordX - destX) ** 2 + (coordY - destY) ** 2)** 0.5

def run_search(image, resolution=250, start=[0,0], end=None, heuristic=euclideanHeuristic, display=False):
    if end==None:
        end = [resolution-1, resolution-1]
    start = [start[1], start[0]]
    end = [end[1], end[0]]
    img = cv2.resize(image, (resolution, resolution))
    plt.imshow(img)
    plt.show()
    #bool_img = [[(lambda x : x[0] == x[1] == x[2] == 0)(p) for p in r] for r in img]
    bool_img = [[(lambda x : x < 0.5)(p) for p in r] for r in img]

    path, found = a_star(bool_img, start, end, heuristic)
    #print(path)
    for i,row in enumerate(img):
        for j, p in enumerate(row):
            img[i][j] = 1 if [i, j] in path else p

    if display:
        plt.imshow(img)
        plt.show()
    return path, found

if __name__ == "__main__":
    path = "images/canny/clean/northeastern_university_17.jpg"
    run_search(cv2.imread(path), resolution=400, start=[100,65], end=[260, 215], heuristic=manhattanHeuristic, display=True)
    