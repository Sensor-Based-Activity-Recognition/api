# from django.shortcuts import render
import io
import json
import gzip
from web.utils.pipeline import Pipeline
from web.utils.modelCNN import Runner as RunnerCNN
from web.utils.modelHGBC import Runner as RunnerHGBC
import pandas as pd

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

runnerCNN = RunnerCNN("web/utils/modelCNN.pkl")
runnerHGBC = RunnerHGBC("web/utils/modelHGBC.pkl")
pipeline = Pipeline()


@csrf_exempt
def index(_):
    return HttpResponse("Use /CNN or /HGBC for API Requests")


# Create your views here.
@csrf_exempt
def CNN(request):
    # decompress the request body
    data = __decompress__(request)

    # run through pipeline
    data = pipeline.run(data, model="CNN")

    # run through model
    result = runnerCNN.run(data)

    # return the result
    return HttpResponse(json.dumps(result))


@csrf_exempt
def HGBC(request):
    # decompress the request body
    data = __decompress__(request)

    # run through pipeline
    data = pipeline.run(data, model="HGBC")

    # run through model
    result = runnerHGBC.run(data)

    # return the result
    return HttpResponse(json.dumps(result))


# Helper functions
def __decompress__(request):
    # decompress the request body
    body = gzip.decompress(request.body)
    # decode the body using utf-8
    body = io.StringIO(body.decode("utf-8"))
    # convert the csv body to a dataframe
    data = pd.read_csv(body)
    return data
