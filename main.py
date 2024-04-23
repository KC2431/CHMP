import torch
import time

class TriangleAlgo:
    """
    Triangle Algorithm
    https://arxiv.org/pdf/1204.1873.pdf

    Code adapted from https://github.com/yikaizhang/AVTA/tree/master
    """
    def __init__(self, setOfPoints, point, eps=1e-3):
        self.setOfPoints = setOfPoints
        self.point = point
        self.eps = eps

    def run(self):
        
        distances = torch.norm(self.setOfPoints - self.point, p=2, dim=1) ** 2
        argMinIndex = torch.argmin(distances)
        pPrime = self.setOfPoints[argMinIndex.item()].unsqueeze(0)
        
        alpha = torch.zeros(self.setOfPoints.shape[0])
        alpha[argMinIndex] = 1
        pIsInOrOut = 1
        
        while torch.norm(self.point - pPrime, p=2) > self.eps:                
            distances = torch.norm(self.point - pPrime, dim=1, p=2)
            gd = torch.matmul(self.setOfPoints, (self.point - pPrime).T)   # (n,dim) (1, dim)
            pointNorm = torch.dot(self.point.squeeze(), self.point.squeeze())
            pPrimeNorm = torch.dot(pPrime.squeeze(), pPrime.squeeze())

            distDiff = (pointNorm - pPrimeNorm) - 2 * gd
            pivotIndex = torch.nonzero(distDiff <= 0).squeeze()
            
            if pivotIndex.nelement() == 0:
                found = 0
            else:
                vIndex = torch.argmax(gd)
                beta = torch.dot((pPrime - self.setOfPoints[vIndex]).squeeze(), (pPrime - self.point).squeeze()) / \
                        torch.dot((pPrime - self.setOfPoints[vIndex]).squeeze(), (pPrime - self.setOfPoints[vIndex]).squeeze())
                alpha = (1 - beta) * alpha
                alpha[vIndex] += beta
                pPrime = (1 - beta) * pPrime + beta * self.setOfPoints[vIndex]
                found = 1
            if found == 0:
                pIsInOrOut = 0
                break

        
        return pPrime, pIsInOrOut
    


class FrankWolfe:
    def __init__(self, point, setOfPoints, eps=1e-3) -> None:
        self.eps = eps
        self.K = 10 / self.eps
        self.point = point
        self.setOfPoints = setOfPoints


    def run(self):
        alpha = torch.zeros(self.setOfPoints.shape[0])
        diff = self.setOfPoints - self.point
        dis = torch.norm(diff, p=2, dim=1) ** 2
        
        minIndex = torch.argmin(dis)
        alpha[minIndex] = 1

        pPrime = self.setOfPoints[minIndex]
        gd = torch.matmul(self.setOfPoints, (self.point - pPrime).T)
        mu = -torch.min(gd)
        Lambda = gd + mu

        for k in range(1, int(self.K)+1):
            gamma = 2 / (2 + self.K)
            if torch.norm(self.point - pPrime, p=2) < self.eps:
                break
            
            s = torch.argmin(gd)
            gamma = 2 / (k + 3)
            alpha = (1 - gamma) * alpha
            alpha[s] += gamma
            pPrime = (1 - gamma) * pPrime + gamma * self.setOfPoints[s]
            gd = torch.matmul(self.setOfPoints, (self.point - pPrime).T)

        return alpha



if __name__ == "__main__":

    torch.manual_seed(23)

    dimensions = [100 * i for i in range(1, 11)]
    K = 500 # Number of Vertices of the convex hull
    n = 10000 # Number of redundant points

    for m in dimensions: 
        convexHullVertices = torch.randn(K, m)
        weights = torch.rand(n, K)
        weights /= weights.sum(dim=1).reshape(-1,1)
        hullData = torch.matmul(weights, convexHullVertices)
        setOfPoints = torch.cat((convexHullVertices, hullData), dim=0)
        
        normalValIndex = torch.arange(n, setOfPoints.shape[0])
        randIndex = torch.randperm(setOfPoints.shape[0])
        
        _, inds = torch.sort(randIndex)
        setOfPoints = setOfPoints[randIndex]
        
        eps = 0.0002 * torch.sqrt(torch.sum(setOfPoints * setOfPoints))

        """
        queryPointWeights = torch.rand(1, K)
        queryPointWeights /= queryPointWeights.sum()
        queryPoint = torch.matmul(queryPointWeights, convexHullVertices)
        """

        queryPoint = torch.randn(1,m)   

        start = time.perf_counter()
        
        frankWolfeAlgo = FrankWolfe(setOfPoints=setOfPoints, point=queryPoint, eps=eps)
        frankWolfeResult = frankWolfeAlgo.run()

        end = time.perf_counter()
        print(f"Total time taken by Frank Wolfe for {m} dimensions: {end-start}")

        start = time.perf_counter()

        triangleAlgo = TriangleAlgo(setOfPoints=setOfPoints, point=queryPoint, eps=eps)
        triangleAlgoResults = triangleAlgo.run()

        end = time.perf_counter()
        print(f"Total time taken by Triangle algorithm for {m} dimensions: {end-start}")

    
    
