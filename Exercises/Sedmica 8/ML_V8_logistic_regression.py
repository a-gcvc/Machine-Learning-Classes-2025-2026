import numpy as np
import math

def logic_score(features, weights, bias):

    score = 0

    for f,w in zip(features, weights):
        score += f*w
    score += bias
    return score


def probbability(z):
    return 1 / (1+math.exp(-z))


def prediction(probabilities):
    prediction = []

    for p in probabilities:
        if p >= 0.5:
            prediction.append(1)
        else:
            prediction.append(0)
            
    return prediction
 