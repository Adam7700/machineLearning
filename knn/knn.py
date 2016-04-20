import math
import operator

""" Converts the originally posted Fisher's
Iris data into usable form"""
def processData(fname,rawData):
    inFile=open(fname)
    #inFile.readline() #No longer needed, stripped from data file
    count=0
    for line in inFile:
        valueList=line.split()
        for i in range(5):
            valueList[i]=float(valueList[i])
        rawData.append(valueList)
        #print(count,rawData[count])
        count+=1

"""Euclidean distance calcs across all features,
but assumes last value is the classification and
so does not include this in the distance calc"""
def euclideanDistance(p1,p2):
    sumDiffSquared=0
    #print(p1,p2)
    for idx in range(len(p1)-1):
        sumDiffSquared+=(p1[idx]-p2[idx])**2
    return round(math.sqrt(sumDiffSquared),3)

""" Convention used is simple distanceMatrix[row][col] where
row=col contains the classification (could be numeric or string)
Only the lower portion of the matrix below (and including the diagonal
with the classification) is created """
def createDistanceMatrix(rawData):
    distanceMatrix=[]
    for row in range(0,len(rawData)):
        mRow=[]
        for col in range(0,row+1):
            if row!=col:
                mRow.append(euclideanDistance(rawData[row],rawData[col]))
            else:
                mRow.append(rawData[row][-1])
        distanceMatrix.append(mRow)
    return distanceMatrix

def classify(point, distMatrix, k, weighted):
    distances = []
    for row in range(0, point):
        distances.append((distMatrix[point][row], distMatrix[row][row]))
    for row in range(point+1, len(distMatrix)):
        distances.append((distMatrix[row][point], distMatrix[row][row]))

    s = sorted(distances, key=lambda x: x[0])

    if weighted == False:
        closest = []
        for i in range(0,k):
            closest.append(s[i][1])

        classification = max(closest, key=closest.count)
    else:
        weights = {0:0, 1:0, 2:0}
        for classNum in s:
            if classNum[0] == 0.0:
                weights[classNum[1]] +=classNum[0]
            else:
                weights[classNum[1]] += 1/float(classNum[0])

        classification = max(weights.iteritems(), key=operator.itemgetter(1))[0]

    return classification

def getErrorRates(errorDict, totalPoints, weighted):
    if weighted:
        print("\nDistance Weighted Error Rates: ")
    else:
        print('Non Distance Weighted Error Rates: ')
    for key in errorDict.keys():
        print(key + ': ' + str(errorDict[key] / float(totalPoints)))

def testK(distanceMatrix, rawData, weighted):
    kError= {'k3':0, 'k5':0, 'k7':0, 'k9':0}
    for point in range(len(distanceMatrix)):
        trueClass = distanceMatrix[point][point]
        print("Point " + str(point) + ': ' + str(rawData[point]))
        for k in range(3,10,2):
            predictedClass = classify(point, distanceMatrix, k, weighted)
            if predictedClass != trueClass:
                kError['k'+str(k)]+=1
            print('k='+str(k)+': '+str(predictedClass))
        print('True class: ' + str(trueClass)+'\n')
    #totalPoints = len(rawData)
    #getErrorRates(kError, totalPoints, weighted)
    return kError

def main():
    rawData=[]
    processData("data.txt",rawData)
    #rawData=[[4.3,3,1.1,0.1,0],[4.8,3,1.4,0.1,0],[4.9,3.1,1.5,0.1,0],[5.2,4.1,1.5,0.1,0],[4.6,3.6,1,0.2,0]] #test with actual data to validate creation of distanceMatrix
    distanceMatrix=createDistanceMatrix(rawData)
    totalPoints= len(rawData)
    #for idx in range(len(distanceMatrix)):
    #    print(idx,distanceMatrix[idx])
    #print('Dist Matrix: ' + str(distanceMatrix))
    nonWeightedError = testK(distanceMatrix, rawData, False)
    weightedError = testK(distanceMatrix, rawData, True)

    getErrorRates(nonWeightedError, totalPoints, False)
    getErrorRates(weightedError, totalPoints, True)







main()
