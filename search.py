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
                return curNode.path
            else: 
                for neighbor in getSuccessors(curNode, map):
                #each neighbor is a (successor,action, stepCost)
                #need to create new Action as a copy!!! otherwise unwanted side effect
                    heapq.heappush(frontier, (len(neighbor.path) + heuristic(neighbor.coordinate, destination), neighbor))

    
    print("PATH NOT FOUND")

def getSuccessors(curNode, map):
    # note x is row y is column
    x, y = curNode.coordinate
    successors = []
    listOfCoords = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]
    for coord in listOfCoords: 
        curX, curY = coord
        if curX >= 0 and curY >= 0 and curX < len(map) and curY < len(map[0]) and map[x][y]:  #it's black
            successors.append(Node(coord, curNode.path + [coord]))
    return successors


def manhattanHeuristic(coord, destination): 
    coordX, coordY = coord
    destX, destY = destination
    return abs(destX - coordX) + abs(destY - coordY)



def euclideanHeuristic(coord, destination): 
    coordX, coordY = coord
    destX, destY = destination
    return ((coordX - destX) ** 2 + (coordY - destY) ** 2)** 0.5

if __name__ == "__main__":
    path = "images/canny/clean/17.jpg"
    img = cv2.resize(cv2.imread(path), (250, 250))
    plt.imshow(img)
    plt.show()
    bool_img = [[(lambda x : x[0] == x[1] == x[2] == 0)(p) for p in r] for r in img]

    path = a_star(bool_img, [30, 60], [69, 160], euclideanHeuristic)
    print(path)
    for i,row in enumerate(img):
        for j, p in enumerate(row):
            img[i][j] = [0, 255, 0] if [i, j] in path else p

    plt.imshow(img)
    plt.show()